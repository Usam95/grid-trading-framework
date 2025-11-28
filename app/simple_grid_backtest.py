# app/simple_grid_backtest.py
from __future__ import annotations

from infra.data_source import LocalFileDataSource
from infra.config_loader import load_run_config
from infra.config_models import RunConfig, GridStrategyConfig, LocalDataConfig
from infra.logging_setup import get_logger, init_logging

from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from core.strategy.grid_strategy_simple import GridConfig, SimpleGridStrategy


def main(config_path: str = "configs/backtests/xrp_grid_1m.yaml") -> None:
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
    if not isinstance(run_cfg.strategy, GridStrategyConfig):
        raise ValueError("Currently only GridStrategyConfig (grid.simple) is supported")

    strat_cfg = run_cfg.strategy

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

    logger.info(
        "Strategy: %s | symbol=%s | base_order_size=%.4f | n_levels=%d | "
        "range_mode=%s | spacing=%s | lower/upper pct=%s/%s | lower/upper price=%s/%s",
        strat_cfg.kind,
        strat_cfg.symbol,
        strat_cfg.base_order_size,
        strat_cfg.n_levels,
        strat_cfg.range_mode.value,
        strat_cfg.spacing.value,
        strat_cfg.lower_pct,
        strat_cfg.upper_pct,
        strat_cfg.lower_price,
        strat_cfg.upper_price,
    )

    # ------------------------------------------------------------------
    # 4) Build low-level BacktestConfig for the engine
    # ------------------------------------------------------------------
    bt_cfg = BacktestConfig(
        symbol=run_cfg.data.symbol,
        start=run_cfg.data.start.isoformat() if run_cfg.data.start else None,
        end=run_cfg.data.end.isoformat() if run_cfg.data.end else None,
        initial_balance=run_cfg.engine.initial_balance,
        trading_fee_pct=run_cfg.engine.trading_fee_pct,
    )

    logger.info(
        "Engine: mode=%s | initial_balance=%.2f | trading_fee_pct=%.6f | slippage_pct=%.6f",
        run_cfg.engine.mode.value,
        run_cfg.engine.initial_balance,
        run_cfg.engine.trading_fee_pct,
        run_cfg.engine.slippage_pct,
    )

    # ------------------------------------------------------------------
    # 5) Run backtest
    # ------------------------------------------------------------------
    engine = BacktestEngine(config=bt_cfg, data_source=ds, strategy=strategy)
    result = engine.run()

    logger.info("Backtest finished. Trades: %d", len(result.trades))
    if result.equity_curve:
        final_ts, final_equity = result.equity_curve[-1]
        logger.info("Final equity at %s: %.2f", final_ts, final_equity)
        print(f"Final equity: {final_equity:.2f}")
    else:
        logger.warning("No equity curve points produced.")

    print(f"Num trades: {len(result.trades)}")

#& "C:\ProgramData\Anaconda3\Scripts\conda.exe" run -n backtester python -m app.simple_grid_backtest
if __name__ == "__main__":
    main()
