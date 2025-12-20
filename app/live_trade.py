# app/live_trade.py
from __future__ import annotations

import os
import signal
import sys
from typing import Optional

from infra.config_loader import load_run_config
from infra.config.engine_config import RunMode
from infra.logging_setup import init_logging, get_logger
from infra.config.binance_live_data_config import BinanceLiveDataConfig
from infra.marketdata.binance_kline_stream import BinanceKlineStream

from infra.secrets import EnvSecretsProvider
from infra.exchange.binance_spot import BinanceSpotExchange


_shutdown_requested = False


def _request_shutdown(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def main(config_path: Optional[str] = None) -> None:
    config_path = config_path or os.environ.get("RUN_CONFIG_PATH", "config/live_trade.yml")

    run_cfg = load_run_config(config_path)

    logfile = init_logging(
        run_name=run_cfg.logging.name,
        level_name=run_cfg.logging.level,
        log_dir=run_cfg.logging.log_dir,
    )
    logger = get_logger(run_cfg.logging.name)

    logger.info("=== Starting LiveTrading runtime (Phase A + Phase B1) ===")
    logger.info("Run: %s", run_cfg.name)
    logger.info("Mode: %s", run_cfg.mode.value)
    logger.info("Config: %s", config_path)
    logger.info("Log file: %s", logfile)

    if not isinstance(run_cfg.data, BinanceLiveDataConfig):
        raise ValueError("For live_trade.py you must set data.source: binance")

    # -------------------------
    # Phase B1: read-only exchange integration (balances + symbol filters)
    # -------------------------
    ex_logger = get_logger("live.exchange")
    try:
        secrets = EnvSecretsProvider()
        exchange = BinanceSpotExchange.from_env(mode=run_cfg.mode, secrets=secrets, logger=ex_logger)
        exchange.connect()
        exchange.ping()

        balances = exchange.get_balances()
        non_zero = [b for b in balances.values() if (abs(b.free) > 0.0 or abs(b.locked) > 0.0)]
        if non_zero:
            ex_logger.info("Account balances (non-zero):")
            for b in sorted(non_zero, key=lambda x: x.asset):
                ex_logger.info("  %s free=%.8f locked=%.8f", b.asset, b.free, b.locked)
        else:
            ex_logger.info("Account balances: all zero (or none returned).")

        filters = exchange.get_symbol_filters(run_cfg.data.symbol)
        ex_logger.info(
            "Symbol filters %s: tick_size=%s step_size=%s min_qty=%s min_notional=%s",
            filters.symbol,
            filters.tick_size,
            filters.step_size,
            filters.min_qty,
            filters.min_notional,
        )
    except Exception as e:
        ex_logger.exception("Exchange read-only init failed (continuing with marketdata only): %s", e)

    # -------------------------
    # Phase A: marketdata stream (closed candles)
    # -------------------------
    inferred_testnet = run_cfg.mode == RunMode.PAPER
    use_testnet_ws = run_cfg.data.use_testnet_ws if run_cfg.data.use_testnet_ws is not None else inferred_testnet

    stream = BinanceKlineStream(
        symbol=run_cfg.data.symbol,
        interval=run_cfg.data.timeframe,
        use_testnet=use_testnet_ws,
        ws_base_url=run_cfg.data.ws_base_url,
        logger=get_logger("live.marketdata"),
    )

    # graceful shutdown
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    stream.start()

    logger.info(
        "Market stream started (symbol=%s interval=%s testnet_ws=%s). Waiting for CLOSED candles...",
        run_cfg.data.symbol,
        run_cfg.data.timeframe,
        use_testnet_ws,
    )

    while not _shutdown_requested:
        try:
            candle = stream.next_closed_candle(timeout_sec=120)
        except TimeoutError:
            logger.warning("No candle received for 120s (still running). Stream status=%s", stream.status())
            continue

        logger.info(
            "CLOSED candle %s O=%.8f H=%.8f L=%.8f C=%.8f V=%.8f",
            candle.timestamp.isoformat(),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )

    logger.info("Shutdown requested. Stopping stream...")
    stream.stop()
    logger.info("Stopped.")


if __name__ == "__main__":
    # optional CLI override:
    #   python -m app.live_trade config/live_trade.yml
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
