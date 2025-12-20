# core/research/grid_search.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
import os
import concurrent.futures as cf

import pandas as pd

from backtest.engine import BacktestEngine
from core.results.models import BacktestResult
from core.results.metrics import create_default_metric_registry

from core.strategy.grid_strategy_simple import (
    GridConfig,
    SimpleGridStrategy,
    GridRangeMode,
    GridSpacing,
)
from core.strategy.grid_strategy_dynamic import (
    DynamicGridConfig,
    DynamicGridStrategy,
)

from infra.config import (
    RunConfig,
    LocalDataConfig,
    BacktestEngineConfig,
    GridStrategyConfig,
    DynamicGridStrategyConfig,
)
from infra.data_source import LocalFileDataSource, InMemoryDataSource, DatasetConfig


# --------------------------------------------------------------------------------------
# Public DTOs
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GridTrainRunResult:
    timeframe: str
    params: Dict[str, Any]
    score: float
    run_id: str
    metrics: Dict[str, float]


@dataclass
class GridResearchSummary:
    started_at: datetime
    finished_at: datetime

    primary_metric: str

    total_combinations: int
    total_timeframes: int
    successful_runs: int
    failed_runs: int

    best_timeframe: str
    best_params: Dict[str, Any]
    best_score: float

    best_train_result: BacktestResult
    best_forward_result: Optional[BacktestResult]

    leaderboard: List[GridTrainRunResult]

    # For saving top-k (app/grid_research.py uses getattr defensively)
    topk_train: List[GridTrainRunResult]
    topk_train_results: List[BacktestResult]

    # Backward-compat alias (older code may still access this)
    top_k_train_results: List[BacktestResult]


# --------------------------------------------------------------------------------------
# Internal worker result (picklable)
# --------------------------------------------------------------------------------------

@dataclass
class _TimeframeEvalResult:
    timeframe: str
    leaderboard: List[GridTrainRunResult]
    top_k_train_results: List[BacktestResult]
    successful_runs: int
    failed_runs: int


def _split_train_forward(df: pd.DataFrame, data_cfg: LocalDataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df according to data_cfg.split:
      - mode='ratio': first train_ratio fraction is train
      - mode='date' : before split_date is train; on/after is forward
    """
    split_cfg = data_cfg.split
    if df.empty:
        return df.copy(), df.copy()

    if split_cfg.mode == "date":
        if split_cfg.split_date is None:
            raise ValueError("split_date must be set when split.mode='date'")
        train = df[df.index < split_cfg.split_date]
        forward = df[df.index >= split_cfg.split_date]
        return train, forward

    # ratio (default)
    n = len(df)
    cut = int(n * split_cfg.train_ratio)
    cut = max(1, min(cut, n - 1))  # keep at least 1 row in each segment
    train = df.iloc[:cut].copy()
    forward = df.iloc[cut:].copy()
    return train, forward


def _candidate_list(values: List[Any], fallback: Any) -> List[Any]:
    return values if values else [fallback]


def _iter_param_combinations(base_strat: Any, grid_cfg: Any) -> List[Dict[str, Any]]:
    """
    Build cartesian product of param grid.

    We grid-search a common subset of fields shared by both GridStrategyConfig and DynamicGridStrategyConfig:
      base_order_size, n_levels, lower_pct, upper_pct, lower_price, upper_price, range_mode, spacing
    """
    base_order_size = _candidate_list(getattr(grid_cfg, "base_order_size", []), getattr(base_strat, "base_order_size"))
    n_levels = _candidate_list(getattr(grid_cfg, "n_levels", []), getattr(base_strat, "n_levels"))

    lower_pct = _candidate_list(getattr(grid_cfg, "lower_pct", []), getattr(base_strat, "lower_pct"))
    upper_pct = _candidate_list(getattr(grid_cfg, "upper_pct", []), getattr(base_strat, "upper_pct"))

    lower_price = _candidate_list(getattr(grid_cfg, "lower_price", []), getattr(base_strat, "lower_price"))
    upper_price = _candidate_list(getattr(grid_cfg, "upper_price", []), getattr(base_strat, "upper_price"))

    range_mode = _candidate_list(getattr(grid_cfg, "range_mode", []), getattr(base_strat, "range_mode"))
    spacing = _candidate_list(getattr(grid_cfg, "spacing", []), getattr(base_strat, "spacing"))

    combos: List[Dict[str, Any]] = []
    for bos, nl, lpct, upct, lpr, upr, rm, sp in product(
        base_order_size, n_levels, lower_pct, upper_pct, lower_price, upper_price, range_mode, spacing
    ):
        combos.append(
            {
                "base_order_size": bos,
                "n_levels": nl,
                "lower_pct": lpct,
                "upper_pct": upct,
                "lower_price": lpr,
                "upper_price": upr,
                "range_mode": rm,
                "spacing": sp,
            }
        )
    return combos


def _build_strategy(strategy_cfg: Any):
    if isinstance(strategy_cfg, GridStrategyConfig):
        cfg = GridConfig(
            symbol=strategy_cfg.symbol,
            base_order_size=strategy_cfg.base_order_size,
            n_levels=strategy_cfg.n_levels,
            range_mode=GridRangeMode(strategy_cfg.range_mode),
            spacing=GridSpacing(strategy_cfg.spacing),
            lower_pct=strategy_cfg.lower_pct,
            upper_pct=strategy_cfg.upper_pct,
            lower_price=strategy_cfg.lower_price,
            upper_price=strategy_cfg.upper_price,
        )
        return SimpleGridStrategy(config=cfg)

    if isinstance(strategy_cfg, DynamicGridStrategyConfig):
        fg = strategy_cfg.floating_grid
        dyn_cfg = DynamicGridConfig(
            symbol=strategy_cfg.symbol,
            base_order_size=strategy_cfg.base_order_size,
            n_levels=strategy_cfg.n_levels,
            range_mode=strategy_cfg.range_mode,
            spacing=strategy_cfg.spacing,
            lower_pct=strategy_cfg.lower_pct,
            upper_pct=strategy_cfg.upper_pct,
            lower_price=strategy_cfg.lower_price,
            upper_price=strategy_cfg.upper_price,
            range_atr_period=strategy_cfg.range_atr_period,
            range_atr_lower_mult=strategy_cfg.range_atr_lower_mult,
            range_atr_upper_mult=strategy_cfg.range_atr_upper_mult,
            use_stop_loss=strategy_cfg.use_stop_loss,
            stop_loss_type=strategy_cfg.stop_loss_type,
            stop_loss_pct=strategy_cfg.stop_loss_pct,
            stop_loss_atr_mult=strategy_cfg.stop_loss_atr_mult,
            use_take_profit=strategy_cfg.use_take_profit,
            take_profit_type=strategy_cfg.take_profit_type,
            take_profit_pct=strategy_cfg.take_profit_pct,
            take_profit_atr_mult=strategy_cfg.take_profit_atr_mult,
            sltp_atr_period=strategy_cfg.sltp_atr_period,
            floating_grid=DynamicGridConfig.FloatingGridConfig(
                enabled=fg.enabled,
                mode=fg.mode,
                band_pct=fg.band_pct,
                use_atr_band=fg.use_atr_band,
                atr_period=fg.atr_period,
                atr_mult=fg.atr_mult,
                interval_days=fg.interval_days,
            ),
            use_rsi_filter=strategy_cfg.use_rsi_filter,
            rsi_period=strategy_cfg.rsi_period,
            rsi_min=strategy_cfg.rsi_min,
            rsi_max=strategy_cfg.rsi_max,
            use_trend_filter=strategy_cfg.use_trend_filter,
            ema_period=strategy_cfg.ema_period,
            max_deviation_pct=strategy_cfg.max_deviation_pct,
        )
        return DynamicGridStrategy(config=dyn_cfg)

    raise ValueError(f"Unsupported strategy config: {type(strategy_cfg)}")


def _push_top_k(top: List[Tuple[float, BacktestResult]], cand_score: float, cand: BacktestResult, k: int) -> None:
    if k <= 0:
        return
    top.append((cand_score, cand))
    top.sort(key=lambda x: x[0], reverse=True)
    del top[k:]


def _worker_eval_timeframe(
    timeframe: str,
    timeframe_index: int,
    total_combinations: int,
    train_df: pd.DataFrame,
    data_cfg_dict: Dict[str, Any],
    engine_cfg_dict: Dict[str, Any],
    strategy_cfg_dict: Dict[str, Any],
    params_list: List[Dict[str, Any]],
    primary_metric: str,
    top_k: int,
    run_name: str,
) -> _TimeframeEvalResult:
    """
    One process evaluates all param combos for ONE timeframe on TRAIN data.
    Also assigns deterministic incrementing run_id:
      run_seq = timeframe_index * total_combinations + param_index + 1
    """
    data_cfg = LocalDataConfig.parse_obj({**data_cfg_dict, "timeframe": timeframe})
    engine_cfg = BacktestEngineConfig.parse_obj(engine_cfg_dict)

    leaderboard: List[GridTrainRunResult] = []
    ok = 0
    fail = 0
    top: List[Tuple[float, BacktestResult]] = []

    for param_index, params in enumerate(params_list):
        try:
            # Deterministic incrementing run_id
            run_seq = timeframe_index * total_combinations + param_index + 1
            run_id_seq = f"{run_seq:06d}"

            # Build strategy config
            if strategy_cfg_dict.get("kind") == "grid.dynamic":
                strat_cfg = DynamicGridStrategyConfig.parse_obj({**strategy_cfg_dict, **params})
            else:
                strat_cfg = GridStrategyConfig.parse_obj({**strategy_cfg_dict, **params})

            strategy = _build_strategy(strat_cfg)

            ds = InMemoryDataSource(train_df)

            engine = BacktestEngine(
                engine_cfg=engine_cfg,
                data_cfg=data_cfg,
                strategy=strategy,
                data_source=ds,
                metric_registry=create_default_metric_registry(),
            )

            result = engine.run()  # NOTE: no kwargs!

            # Preserve engine UUID for debugging, but expose seq run_id everywhere
            engine_uuid = result.run_id
            result.extra.setdefault("run_ids", {})
            result.extra["run_ids"].update({"engine_uuid": engine_uuid, "research_seq": run_id_seq})
            result.run_id = run_id_seq

            # annotate for traceability
            result.run_name = f"{run_name}:train:{timeframe}"
            result.extra.setdefault("research", {})
            result.extra["research"].update(
                {
                    "segment": "train",
                    "timeframe": timeframe,
                    "params": params,
                    "primary_metric": primary_metric,
                }
            )

            score = float(result.metrics.get(primary_metric, 0.0))
            leaderboard.append(
                GridTrainRunResult(
                    timeframe=timeframe,
                    params=params,
                    score=score,
                    run_id=result.run_id,
                    metrics=result.metrics,
                )
            )

            _push_top_k(top, score, result, top_k)
            ok += 1

        except Exception:
            fail += 1

    leaderboard.sort(key=lambda r: r.score, reverse=True)
    top_results = [r for _, r in sorted(top, key=lambda x: x[0], reverse=True)]

    return _TimeframeEvalResult(
        timeframe=timeframe,
        leaderboard=leaderboard,
        top_k_train_results=top_results,
        successful_runs=ok,
        failed_runs=fail,
    )


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

class GridResearchRunner:
    def __init__(self, run_cfg: RunConfig):
        if run_cfg.research is None or not run_cfg.research.enabled:
            raise ValueError("RunConfig.research must be present and enabled for GridResearchRunner.")

        self.run_cfg = run_cfg
        self.research_cfg = run_cfg.research

        # Data source for loading raw history
        self.file_ds = LocalFileDataSource()

    def run(self) -> GridResearchSummary:
        started_at = datetime.now()

        data_cfg_base: LocalDataConfig = self.run_cfg.data
        engine_cfg: BacktestEngineConfig = self.run_cfg.engine

        timeframes = data_cfg_base.research_timeframes or [data_cfg_base.timeframe]

        # Load & split datasets (main process)
        tf_train: Dict[str, pd.DataFrame] = {}
        tf_forward: Dict[str, pd.DataFrame] = {}

        for tf in timeframes:
            ds_cfg = DatasetConfig(
                symbol=data_cfg_base.symbol,
                start=data_cfg_base.start,
                end=data_cfg_base.end,
                timeframe=tf,
            )
            df = self.file_ds.load(ds_cfg)
            train_df, forward_df = _split_train_forward(df, data_cfg_base)
            tf_train[tf] = train_df
            tf_forward[tf] = forward_df

        # Param combinations (grid.simple OR grid.dynamic)
        if not isinstance(self.run_cfg.strategy, (GridStrategyConfig, DynamicGridStrategyConfig)):
            raise ValueError("GridResearchRunner supports grid.simple and grid.dynamic only.")

        all_params = _iter_param_combinations(self.run_cfg.strategy, self.research_cfg.param_grid)

        if self.research_cfg.max_runs is not None:
            all_params = all_params[: self.research_cfg.max_runs]

        total_combinations = len(all_params)
        primary_metric = self.research_cfg.primary_metric
        top_k = getattr(self.research_cfg, "top_k", 5)

        # Multiprocessing over timeframes
        max_workers_cfg = getattr(self.research_cfg, "max_workers", None)
        if max_workers_cfg is None:
            cpu = os.cpu_count() or 1
            max_workers_cfg = max(1, min(len(timeframes), cpu))

        # serialize configs for workers
        data_cfg_dict = data_cfg_base.dict()
        engine_cfg_dict = engine_cfg.dict()
        strategy_cfg_dict = self.run_cfg.strategy.dict()

        results: List[_TimeframeEvalResult] = []

        if max_workers_cfg <= 1 or len(timeframes) <= 1:
            for tf_idx, tf in enumerate(timeframes):
                results.append(
                    _worker_eval_timeframe(
                        timeframe=tf,
                        timeframe_index=tf_idx,
                        total_combinations=total_combinations,
                        train_df=tf_train[tf],
                        data_cfg_dict=data_cfg_dict,
                        engine_cfg_dict=engine_cfg_dict,
                        strategy_cfg_dict=strategy_cfg_dict,
                        params_list=all_params,
                        primary_metric=primary_metric,
                        top_k=top_k,
                        run_name=self.run_cfg.name,
                    )
                )
        else:
            with cf.ProcessPoolExecutor(max_workers=max_workers_cfg) as ex:
                futs = [
                    ex.submit(
                        _worker_eval_timeframe,
                        tf,
                        tf_idx,
                        total_combinations,
                        tf_train[tf],
                        data_cfg_dict,
                        engine_cfg_dict,
                        strategy_cfg_dict,
                        all_params,
                        primary_metric,
                        top_k,
                        self.run_cfg.name,
                    )
                    for tf_idx, tf in enumerate(timeframes)
                ]
                for fut in cf.as_completed(futs):
                    results.append(fut.result())

        # Merge leaderboards + topk candidates
        merged_lb: List[GridTrainRunResult] = []
        topk_candidates: List[BacktestResult] = []
        ok = 0
        fail = 0

        for r in results:
            merged_lb.extend(r.leaderboard)
            topk_candidates.extend(r.top_k_train_results)
            ok += r.successful_runs
            fail += r.failed_runs

        merged_lb.sort(key=lambda x: x.score, reverse=True)
        if not merged_lb:
            raise RuntimeError("All research runs failed; no leaderboard entries produced.")

        best_entry = merged_lb[0]
        best_timeframe = best_entry.timeframe
        best_params = best_entry.params
        best_score = best_entry.score

        # Best train result object: find inside topk_candidates first; if not present, rerun once
        best_train_result = next((x for x in topk_candidates if x.run_id == best_entry.run_id), None)
        if best_train_result is None:
            best_train_result = self._run_single(
                best_timeframe, tf_train[best_timeframe], best_params, segment="train", forced_run_id=best_entry.run_id
            )

        # Forward run (main process) on best timeframe (use SAME numeric run_id; segment differentiates)
        best_forward_result = None
        forward_df = tf_forward.get(best_timeframe)
        if forward_df is not None and not forward_df.empty:
            best_forward_result = self._run_single(
                best_timeframe, forward_df, best_params, segment="forward", forced_run_id=best_entry.run_id
            )

        # Global top-k (full results) for saving
        topk_candidates.sort(key=lambda r: float(r.metrics.get(primary_metric, 0.0)), reverse=True)
        top_k_train_results = topk_candidates[:top_k] if top_k > 0 else []

        # Build aligned "topk_train" entries for app saving
        by_id: Dict[str, GridTrainRunResult] = {e.run_id: e for e in merged_lb}
        topk_train_entries: List[GridTrainRunResult] = [by_id[r.run_id] for r in top_k_train_results if r.run_id in by_id]

        finished_at = datetime.now()

        return GridResearchSummary(
            started_at=started_at,
            finished_at=finished_at,
            primary_metric=primary_metric,
            total_combinations=total_combinations,
            total_timeframes=len(timeframes),
            successful_runs=ok,
            failed_runs=fail,
            best_timeframe=best_timeframe,
            best_params=best_params,
            best_score=best_score,
            best_train_result=best_train_result,
            best_forward_result=best_forward_result,
            leaderboard=merged_lb,
            topk_train=topk_train_entries,
            topk_train_results=top_k_train_results,
            top_k_train_results=top_k_train_results,
        )

    def _run_single(
        self,
        timeframe: str,
        df: pd.DataFrame,
        params: Dict[str, Any],
        segment: str,
        forced_run_id: Optional[str] = None,
    ) -> BacktestResult:
        data_cfg = self.run_cfg.data.copy()
        data_cfg.timeframe = timeframe

        if self.run_cfg.strategy.kind == "grid.dynamic":
            strat_cfg = DynamicGridStrategyConfig.parse_obj({**self.run_cfg.strategy.dict(), **params})
        else:
            strat_cfg = GridStrategyConfig.parse_obj({**self.run_cfg.strategy.dict(), **params})

        strategy = _build_strategy(strat_cfg)

        engine = BacktestEngine(
            engine_cfg=self.run_cfg.engine,
            data_cfg=data_cfg,
            strategy=strategy,
            data_source=InMemoryDataSource(df),
            metric_registry=create_default_metric_registry(),
        )

        result = engine.run()  # NOTE: no kwargs
        engine_uuid = result.run_id

        if forced_run_id is not None:
            result.extra.setdefault("run_ids", {})
            result.extra["run_ids"].update({"engine_uuid": engine_uuid, "research_seq": forced_run_id})
            result.run_id = forced_run_id

        result.run_name = f"{self.run_cfg.name}:{segment}:{timeframe}"
        result.extra.setdefault("research", {})
        result.extra["research"].update(
            {
                "segment": segment,
                "timeframe": timeframe,
                "params": params,
                "primary_metric": self.research_cfg.primary_metric,
            }
        )
        return result
