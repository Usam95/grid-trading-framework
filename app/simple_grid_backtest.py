from __future__ import annotations

import time
from pathlib import Path

from infra.data_source import LocalFileDataSource
from infra.config_loader import load_run_config
from infra.config import RunConfig, GridStrategyConfig, LocalDataConfig
from infra.config.strategy_grid import DynamicGridStrategyConfig
from infra.logging_setup import get_logger, init_logging

from backtest.engine import BacktestEngine
from core.strategy.grid_strategy_simple import GridConfig, SimpleGridStrategy
from core.strategy.grid_strategy_dynamic import DynamicGridConfig, DynamicGridStrategy

from core.results.metrics import create_default_metric_registry
from core.results.repository import save_backtest_result
from core.results.summary import print_result_summary


def main(config_path: str = "config/grid_run.yml") -> None:
    # ------------------------------------------------------------------
    # 1) Load high-level run configuration
    # ------------------------------------------------------------------
    run_cfg: RunConfig = load_run_config(config_path)

    # --- INIT LOGGING HERE ---
    logfile = init_logging(
        run_name=run_cfg.logging.name,
        level_name=run_cfg.logging.level,
        log_dir=run_cfg.logging.log_dir,
    )
    logger = get_logger(run_cfg.logging.name)
    logger.info("=== Starting backtest run: %s ===", run_cfg.name)
    logger.info("Description: %s", run_cfg.description or "(none)")
    logger.info("Log file for this run: %s", logfile)

    # ------------------------------------------------------------------
    # 2) Build data source from config
    # ------------------------------------------------------------------
    if not isinstance(run_cfg.data, LocalDataConfig):
        raise ValueError("Currently only LocalDataConfig is supported for data.source")

    ds = LocalFileDataSource(root=run_cfg.data.root)
    logger.info(
        "Data source: local, symbol=%s, start=%s, end=%s, root=%s",
        run_cfg.data.symbol,
        run_cfg.data.start,
        run_cfg.data.end,
        run_cfg.data.root,
    )

    # ------------------------------------------------------------------
    # 3) Build strategy from config
    # ------------------------------------------------------------------
    strat_cfg = run_cfg.strategy

    if isinstance(strat_cfg, GridStrategyConfig):
        # Simple static grid
        grid_cfg = GridConfig(
            symbol=strat_cfg.symbol,
            base_order_size=strat_cfg.base_order_size,
            n_levels=strat_cfg.n_levels,
            lower_pct=strat_cfg.lower_pct,
            upper_pct=strat_cfg.upper_pct,
            lower_price=strat_cfg.lower_price,
            upper_price=strat_cfg.upper_price,
            range_mode=strat_cfg.range_mode,
            spacing=strat_cfg.spacing,
        )
        strategy = SimpleGridStrategy(config=grid_cfg)

    elif isinstance(strat_cfg, DynamicGridStrategyConfig):
        # Dynamic / advanced grid
        dyn_cfg = DynamicGridConfig(
            symbol=strat_cfg.symbol,
            base_order_size=strat_cfg.base_order_size,
            n_levels=strat_cfg.n_levels,
            lower_pct=strat_cfg.lower_pct,
            upper_pct=strat_cfg.upper_pct,
            lower_price=strat_cfg.lower_price,
            upper_price=strat_cfg.upper_price,
            range_mode=strat_cfg.range_mode,
            spacing=strat_cfg.spacing,
            spacing_mode=strat_cfg.spacing_mode,
            spacing_pct=strat_cfg.spacing_pct,
            spacing_atr_mult=strat_cfg.spacing_atr_mult,
            range_atr_lower_mult=strat_cfg.range_atr_lower_mult,
            range_atr_upper_mult=strat_cfg.range_atr_upper_mult,
            use_stop_loss=strat_cfg.use_stop_loss,
            stop_loss_type=strat_cfg.stop_loss_type,
            stop_loss_pct=strat_cfg.stop_loss_pct,
            stop_loss_atr_mult=strat_cfg.stop_loss_atr_mult,
            use_take_profit=strat_cfg.use_take_profit,
            take_profit_type=strat_cfg.take_profit_type,
            take_profit_pct=strat_cfg.take_profit_pct,
            take_profit_atr_mult=strat_cfg.take_profit_atr_mult,
            recenter_mode=strat_cfg.recenter_mode,
            recenter_band_pct=strat_cfg.recenter_band_pct,
            recenter_atr_mult=strat_cfg.recenter_atr_mult,
            recenter_interval_days=strat_cfg.recenter_interval_days,
            use_rsi_filter=strat_cfg.use_rsi_filter,
            rsi_period=strat_cfg.rsi_period,
            rsi_min=strat_cfg.rsi_min,
            rsi_max=strat_cfg.rsi_max,
            use_trend_filter=strat_cfg.use_trend_filter,
            ema_period=strat_cfg.ema_period,
            max_deviation_pct=strat_cfg.max_deviation_pct,
        )
        strategy = DynamicGridStrategy(config=dyn_cfg)

    else:
        raise ValueError(f"Unsupported strategy config type: {type(strat_cfg)}")

    logger.info("Strategy kind=%s", strat_cfg.kind)
    logger.info(
        "Strategy: %s | symbol=%s | base_order_size=%.4f | n_levels=%d | "
        "range_mode=%s | spacing=%s | lower/upper pct=%s/%s | lower/upper price=%s/%s",
        strat_cfg.kind,
        strat_cfg.symbol,
        strat_cfg.base_order_size,
        strat_cfg.n_levels,
        strat_cfg.range_mode.value if hasattr(strat_cfg.range_mode, "value") else strat_cfg.range_mode,
        strat_cfg.spacing.value if hasattr(strat_cfg.spacing, "value") else strat_cfg.spacing,
        strat_cfg.lower_pct,
        strat_cfg.upper_pct,
        strat_cfg.lower_price,
        strat_cfg.upper_price,
    )

    # ------------------------------------------------------------------
    # 4) Log engine config (BacktestEngineConfig)
    # ------------------------------------------------------------------
    logger.info(
        "Engine: mode=%s | initial_balance=%.2f | trading_fee_pct=%.6f | "
        "slippage_pct=%.6f | max_candles=%s",
        run_cfg.engine.mode.value,
        run_cfg.engine.initial_balance,
        run_cfg.engine.trading_fee_pct,
        run_cfg.engine.slippage_pct,
        str(run_cfg.engine.max_candles),
    )
    logger.info("Engine metrics: %s", run_cfg.engine.metrics)

    # ------------------------------------------------------------------
    # 5) Build metric registry and engine
    # ------------------------------------------------------------------
    metric_registry = create_default_metric_registry()

    engine = BacktestEngine(
        engine_cfg=run_cfg.engine,
        data_cfg=run_cfg.data,
        strategy=strategy,
        data_source=ds,
        metric_registry=metric_registry,
    )

    # ------------------------------------------------------------------
    # 6) Run backtest (measure execution time)
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    result = engine.run()
    t_end = time.perf_counter()
    elapsed_sec = t_end - t_start
    elapsed_min = elapsed_sec / 60.0

    logger.info(
        "Backtest execution time: %.2f seconds (%.2f minutes)",
        elapsed_sec,
        elapsed_min,
    )

    # ------------------------------------------------------------------
    # 7) Save results & print summary
    # ------------------------------------------------------------------
    symbol = run_cfg.data.symbol
    run_dir = save_backtest_result(
        result,
        base_dir=Path("output") / symbol / "manual_runs",
    )

    logger.info("Results saved to %s", run_dir)
    print_result_summary(result, logger)

    print(f"Final equity:   {result.final_equity:.2f}")
    print(f"Num trades:     {len(result.trades)}")
    print(f"Run time:       {elapsed_sec:.2f} s ({elapsed_min:.2f} min)")


if __name__ == "__main__":
    main()
