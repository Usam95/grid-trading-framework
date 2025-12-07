# core/research/grid_search.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from backtest.engine import BacktestEngine
from core.results.metrics import create_default_metric_registry, MetricRegistry
from core.results.models import BacktestResult
from core.strategy.grid_strategy_simple import (
    GridConfig,
    SimpleGridStrategy,
    GridRangeMode,
    GridSpacing,
)
from infra.config.data_config import LocalDataConfig
from infra.config.research_config import GridResearchConfig, GridParamGridConfig
from infra.config.run_config import RunConfig
from infra.data_source import LocalFileDataSource, DatasetConfig, InMemoryDataSource
from infra.logging_setup import get_logger
from infra.splits import split_train_forward


log = get_logger("research.grid")


@dataclass
class GridTrainRunResult:
    """
    Single train-backtest result for one parameter combination.
    """
    timeframe: str
    params: Dict[str, Any]
    score: float  
    metrics: Dict[str, float]
    run_id: str


@dataclass
class GridResearchSummary:
    """
    Aggregate outcome of a full grid research run.
    """
    leaderboard: List[GridTrainRunResult]
    best_timeframe: str
    best_params: Dict[str, Any]
    best_train_result: BacktestResult
    best_forward_result: BacktestResult


class GridResearchRunner:
    """
    Orchestrates parameter search for the simple grid strategy.

    Responsibilities:
      - Load and resample data for all requested timeframes.
      - Split each timeframe into train / forward using DataSplitConfig.
      - Enumerate parameter combinations from GridParamGridConfig.
      - Run backtests on train data in-memory (no repeated disk I/O).
      - Score each run with grid_objective().
      - Select best combo and run a forward backtest with the same params.
    """

    def __init__(self, run_cfg: RunConfig) -> None:
        self.run_cfg = run_cfg
        self.data_cfg: LocalDataConfig = run_cfg.data  # type: ignore
        self.research_cfg: GridResearchConfig | None = run_cfg.research  # type: ignore
        self.metric_registry: MetricRegistry = create_default_metric_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> GridResearchSummary:
        if not self.research_cfg or not self.research_cfg.enabled:
            raise ValueError("Research is not enabled in the run config (research.enabled=false).")

        if self.research_cfg.param_grid.is_empty():
            raise ValueError("Research param_grid is empty; configure at least one parameter list.")

        # Determine which timeframes to include in the search
        timeframes = self._collect_timeframes()
        log.info("Grid research timeframes: %s", timeframes)

        # Load & split data once per timeframe
        tf_datasets = self._prepare_datasets(timeframes)

        leaderboard: List[GridTrainRunResult] = []
        best_score = float("-inf")
        best_params: Dict[str, Any] | None = None
        best_timeframe: str | None = None
        best_train_result: BacktestResult | None = None
        best_forward_result: BacktestResult | None = None

        max_runs = self.research_cfg.max_runs
        run_counter = 0

        for timeframe in timeframes:
            train_df, forward_df = tf_datasets[timeframe]

            for params in self._iter_param_combinations(self.research_cfg.param_grid, timeframe):
                if max_runs is not None and run_counter >= max_runs:
                    log.info("Reached max_runs=%d, stopping parameter search.", max_runs)
                    break

                run_counter += 1
                log.info(
                    "=== Research run %d (timeframe=%s, params=%s) ===",
                    run_counter,
                    timeframe,
                    params,
                )

                # Train run
                train_result = self._run_single_backtest(
                    timeframe=timeframe,
                    df=train_df,
                    params=params,
                    segment="train",
                )

                primary = self.research_cfg.primary_metric
                score = train_result.metrics.get(primary, 0.0)

                leaderboard.append(
                    GridTrainRunResult(
                        timeframe=timeframe,
                        params=params,
                        score=score,
                        metrics=train_result.metrics,
                        run_id=train_result.run_id,
                    )
                )

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_timeframe = timeframe
                    best_train_result = train_result
                    log.debug(
                        "New best config: score=%.4f timeframe=%s params=%s",
                        best_score,
                        best_timeframe,
                        best_params,
                    )

            if max_runs is not None and run_counter >= max_runs:
                break

        if not leaderboard or best_params is None or best_timeframe is None or best_train_result is None:
            raise RuntimeError("No successful research runs were executed.")

        # Forward run with best config on corresponding forward data
        best_train_df, best_forward_df = tf_datasets[best_timeframe]
        best_forward_result = self._run_single_backtest(
            timeframe=best_timeframe,
            df=best_forward_df,
            params=best_params,
            segment="forward",
        )

        return GridResearchSummary(
            leaderboard=leaderboard,
            best_timeframe=best_timeframe,
            best_params=best_params,
            best_train_result=best_train_result,
            best_forward_result=best_forward_result,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_timeframes(self) -> List[str]:
        """
        Use the base data timeframe plus any research_timeframes
        (without duplicates, base timeframe first).
        """
        base_tf = self.data_cfg.timeframe
        tfs: List[str] = []

        if base_tf:
            tfs.append(base_tf)

        for tf in self.data_cfg.research_timeframes:
            if tf not in tfs:
                tfs.append(tf)

        if not tfs:
            raise ValueError("No timeframes configured in data.timeframe/research_timeframes.")
        return tfs

    def _prepare_datasets(
        self,
        timeframes: List[str],
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        For each timeframe:
          - load full dataset via LocalFileDataSource
          - resample to timeframe
          - split into train / forward via DataSplitConfig
        """
        ds = LocalFileDataSource(root=self.data_cfg.root)
        out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

        for tf in timeframes:
            cfg = DatasetConfig(
                symbol=self.data_cfg.symbol,
                start=self.data_cfg.start,
                end=self.data_cfg.end,
                timeframe=tf,
            )
            full_df = ds.load(cfg)

            train_df, forward_df = split_train_forward(full_df, self.data_cfg.split)
            log.info(
                "Prepared timeframe=%s: %d rows total -> %d train, %d forward.",
                tf,
                len(full_df),
                len(train_df),
                len(forward_df),
            )
            out[tf] = (train_df, forward_df)

        return out

    def _iter_param_combinations(
        self,
        grid: GridParamGridConfig,
        timeframe: str,
    ) -> Dict[str, Any]:
        """
        Generate parameter combinations. For each field:
          - if the grid list is non-empty, use it
          - otherwise, fall back to the base strategy config value
        """
        base: GridStrategyConfig = self.run_cfg.strategy  # type: ignore

        cand_base_order_size = grid.base_order_size or [base.base_order_size]
        cand_n_levels = grid.n_levels or [base.n_levels]
        cand_lower_pct = grid.lower_pct or [base.lower_pct]
        cand_upper_pct = grid.upper_pct or [base.upper_pct]
        cand_lower_price = grid.lower_price or [base.lower_price]
        cand_upper_price = grid.upper_price or [base.upper_price]
        cand_range_mode = grid.range_mode or [base.range_mode]
        cand_spacing = grid.spacing or [base.spacing]

        for (
            bos,
            nlv,
            lpct,
            upct,
            lpr,
            upr,
            rmode,
            spacing,
        ) in product(
            cand_base_order_size,
            cand_n_levels,
            cand_lower_pct,
            cand_upper_pct,
            cand_lower_price,
            cand_upper_price,
            cand_range_mode,
            cand_spacing,
        ):
            yield {
                "timeframe": timeframe,
                "base_order_size": bos,
                "n_levels": nlv,
                "lower_pct": lpct,
                "upper_pct": upct,
                "lower_price": lpr,
                "upper_price": upr,
                "range_mode": rmode,
                "spacing": spacing,
            }

    def _build_grid_strategy(self, params: Dict[str, Any]) -> SimpleGridStrategy:
        """
        Build a SimpleGridStrategy instance from base strategy config
        plus parameter overrides from the grid.
        """
        base: GridStrategyConfig = self.run_cfg.strategy  # type: ignore

        range_mode = params["range_mode"]
        spacing = params["spacing"]

        # Ensure we always have proper enums
        if isinstance(range_mode, str):
            range_mode = GridRangeMode(range_mode.lower())
        if isinstance(spacing, str):
            spacing = GridSpacing(spacing.lower())

        grid_cfg = GridConfig(
            symbol=base.symbol,
            base_order_size=float(params["base_order_size"]),
            n_levels=int(params["n_levels"]),
            lower_pct=params["lower_pct"],
            upper_pct=params["upper_pct"],
            lower_price=params["lower_price"],
            upper_price=params["upper_price"],
            range_mode=range_mode,
            spacing=spacing,
        )

        return SimpleGridStrategy(config=grid_cfg)

    def _run_single_backtest(
        self,
        timeframe: str,
        df: pd.DataFrame,
        params: Dict[str, Any],
        segment: str,
    ) -> BacktestResult:
        """
        Run a single backtest on an in-memory dataset for the given params.
        """
        # Clone data config with updated timeframe so logs & result are consistent
        data_cfg_for_tf: LocalDataConfig = self.data_cfg.copy(update={"timeframe": timeframe})

        strategy = self._build_grid_strategy(params)
        data_source = InMemoryDataSource(df)

        engine = BacktestEngine(
            engine_cfg=self.run_cfg.engine,
            data_cfg=data_cfg_for_tf,
            strategy=strategy,
            data_source=data_source,
            metric_registry=self.metric_registry,
        )

        log.info(
            "Running %s backtest: timeframe=%s, symbol=%s, params=%s",
            segment,
            timeframe,
            data_cfg_for_tf.symbol,
            params,
        )

        result = engine.run()

        log.info(
            "%s backtest completed: final_equity=%.2f, total_return_pct=%.2f, "
            "max_drawdown_pct=%.2f, trades=%d",
            segment,
            result.final_equity,
            result.metrics.get("total_return_pct", 0.0),
            result.metrics.get("max_drawdown_pct", 0.0),
            int(result.metrics.get("n_trades", 0.0)),
        )
        return result
