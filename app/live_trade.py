# app/live_trade.py
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Optional, List

try:
    import yaml  # type: ignore
except ImportError as e:
    raise ImportError("Missing dependency: PyYAML. Install via: pip install pyyaml") from e

from infra.logging_setup import init_logging, get_logger
from infra.config.engine_config import RunMode
from infra.secrets import EnvSecretsProvider

from infra.exchange.binance_spot import BinanceSpotExchange
from infra.marketdata.binance_kline_stream import BinanceKlineStream
from infra.marketdata.binance_user_stream import BinanceUserDataStream, UserStreamConfig


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

@dataclass
class LiveTradeConfig:
    run_name: str
    mode: RunMode
    config_path: str
    log_level: str

    symbol: str
    interval: str

    # If True: require secrets and private endpoints (balances/open orders/user stream/B3/B5).
    # If False: allow public-only run (ping/server_time + kline stream).
    require_private_endpoints: bool = True

    # B3
    b3_enabled: bool = False
    b3_side: str = "BUY"
    b3_qty: float = 0.1
    b3_wait_fills_sec: int = 20

    # B4
    b4_enabled: bool = True
    user_keepalive_sec: int = 30 * 60
    user_reconnect_backoff_sec: int = 5

    # B5 (LIMIT lifecycle test)
    b5_enabled: bool = True
    b5_side: str = "BUY"
    b5_qty: float = 5.0
    b5_price_offset_pct: float = 0.01     # BUY: 1% below market, SELL: 1% above
    b5_cancel_after_sec: int = 5

    # logging hygiene
    balances_log_assets: Optional[List[str]] = None


def _parse_mode(raw_mode: str) -> RunMode:
    m = (raw_mode or "").strip().lower()
    if m in {"paper", "testnet"}:
        return RunMode.PAPER
    if m in {"live", "prod", "production", "real"}:
        return RunMode.LIVE
    return RunMode.PAPER


def _mask_secret(s: str, keep: int = 4) -> str:
    if not s:
        return "<EMPTY>"
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)


def load_config(path: str) -> LiveTradeConfig:
    """
    Reads config/test_live_trading.yml (RunConfig-like) without duplicating config in code.
    Also supports your legacy shape for backward compatibility.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Preferred structure: name/engine/data/logging/live
    name = raw.get("name") or raw.get("run_name") or "xrp_live_phase_b"
    engine = raw.get("engine") if isinstance(raw.get("engine"), dict) else {}
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    logging = raw.get("logging") if isinstance(raw.get("logging"), dict) else {}
    live = raw.get("live") if isinstance(raw.get("live"), dict) else {}

    # mode
    mode_raw = engine.get("mode") or raw.get("mode") or "paper"
    mode = _parse_mode(str(mode_raw))

    # market data params
    symbol = data.get("symbol") or raw.get("symbol") or "XRPUSDT"
    interval = data.get("timeframe") or raw.get("interval") or "1m"

    # logging
    log_level = str(logging.get("level") or raw.get("log_level") or "INFO")

    # private endpoints gate
    require_private = bool(live.get("require_private_endpoints", raw.get("require_private_endpoints", True)))

    # B3 (new keys in live section)
    # (legacy fallback: b3_test_order.{enabled,side,qty,wait_fills_sec})
    b3_legacy = raw.get("b3_test_order") if isinstance(raw.get("b3_test_order"), dict) else {}
    b3_enabled = bool(live.get("b3_enabled", b3_legacy.get("enabled", False)))
    b3_side = str(live.get("b3_side", b3_legacy.get("side", "BUY"))).upper()
    b3_qty = float(live.get("b3_qty", b3_legacy.get("qty", 0.1)))
    b3_wait = int(live.get("b3_wait_fills_sec", b3_legacy.get("wait_fills_sec", 20)))

    # B4 (new keys in live section)
    # (legacy fallback: b4_user_stream.{enabled,keepalive_interval_sec,reconnect_backoff_sec})
    b4_legacy = raw.get("b4_user_stream") if isinstance(raw.get("b4_user_stream"), dict) else {}
    b4_enabled = bool(live.get("b4_enabled", b4_legacy.get("enabled", True)))
    keepalive_sec = int(live.get("user_keepalive_sec", b4_legacy.get("keepalive_interval_sec", 30 * 60)))
    reconnect_backoff_sec = int(live.get("user_reconnect_backoff_sec", b4_legacy.get("reconnect_backoff_sec", 5)))

    # B5
    b5_enabled = bool(live.get("b5_enabled", False))
    b5_side = str(live.get("b5_side", "BUY")).upper()
    b5_qty = float(live.get("b5_qty", 5.0))
    b5_price_offset_pct = float(live.get("b5_price_offset_pct", 0.01))
    b5_cancel_after_sec = int(live.get("b5_cancel_after_sec", 5))

    balances_log_assets = live.get("balances_log_assets", None)
    if balances_log_assets is not None and not isinstance(balances_log_assets, list):
        balances_log_assets = None

    return LiveTradeConfig(
        run_name=name,
        mode=mode,
        config_path=path,
        log_level=log_level,
        symbol=symbol,
        interval=interval,
        require_private_endpoints=require_private,
        b3_enabled=b3_enabled,
        b3_side=b3_side,
        b3_qty=b3_qty,
        b3_wait_fills_sec=b3_wait,
        b4_enabled=b4_enabled,
        user_keepalive_sec=keepalive_sec,
        user_reconnect_backoff_sec=reconnect_backoff_sec,
        b5_enabled=b5_enabled,
        b5_side=b5_side,
        b5_qty=b5_qty,
        b5_price_offset_pct=b5_price_offset_pct,
        b5_cancel_after_sec=b5_cancel_after_sec,
        balances_log_assets=balances_log_assets,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/test_live_trading.yml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_file = init_logging(run_name=cfg.run_name, level_name=cfg.log_level)
    log = get_logger("live")
    ex_log = get_logger("live.exchange")
    md_log = get_logger("live.marketdata")
    us_log = get_logger("live.userstream")

    log.info("=== Starting LiveTrading runtime (B1..B5 test harness) ===")
    log.info("Run: %s", cfg.run_name)
    log.info("Mode: %s", cfg.mode.value)
    log.info("Config: %s", cfg.config_path)
    log.info("Log level: %s", cfg.log_level)
    log.info("Log file: %s", str(log_file))
    log.info("Symbol: %s | Interval: %s", cfg.symbol, cfg.interval)

    # -----------------------------------------------------------------------
    # Secrets: ALWAYS from infra/secrets.py (EnvSecretsProvider)
    # -----------------------------------------------------------------------
    secrets = EnvSecretsProvider()

    api_key = ""
    api_secret = ""
    try:
        api_key, api_secret = secrets.get_binance_keys(cfg.mode)
        log.info("Resolved API key from env: %s", _mask_secret(api_key, keep=6))
        log.info("Resolved API secret from env: %s", _mask_secret(api_secret, keep=2))
    except Exception as e:
        if cfg.require_private_endpoints:
            raise RuntimeError(
                f"Missing API credentials for private endpoints via EnvSecretsProvider: {e}. "
                f"For PAPER set BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET. "
                f"For LIVE set BINANCE_API_KEY / BINANCE_API_SECRET."
            ) from e
        log.warning(
            "Secrets not available (%s). Continuing PUBLIC-ONLY mode (no balances/open orders/user stream/orders).",
            e,
        )

    has_private_creds = bool(api_key) and bool(api_secret)

    # -----------------------------------------------------------------------
    # Exchange bootstrap (B1 foundation)
    # -----------------------------------------------------------------------
    exchange = BinanceSpotExchange(
        mode=cfg.mode,
        api_key=api_key,
        api_secret=api_secret,
        logger_name="live.exchange",
    )

    exchange.ping()
    st = exchange.server_time()
    ex_log.info("Binance server time: %s", st)

    # -----------------------------------------------------------------------
    # Private endpoints (B2/B4/B3/B5) only when creds exist
    # -----------------------------------------------------------------------
    exec_reports: "Queue[dict[str, Any]]" = Queue()
    user_stream: Optional[BinanceUserDataStream] = None

    filt = None

    if has_private_creds:
        # balances
        balances = exchange.get_balances(non_zero_only=True)
        log.info("Account balances (non-zero):")

        if cfg.balances_log_assets:
            for a in cfg.balances_log_assets:
                if a in balances:
                    free, locked = balances[a]
                    log.info("  %s free=%.8f locked=%.8f", a, free, locked)
        else:
            for asset in sorted(balances.keys()):
                free, locked = balances[asset]
                log.info("  %s free=%.8f locked=%.8f", asset, free, locked)

        # symbol filters
        filt = exchange.get_symbol_filters(cfg.symbol)
        log.info(
            "Symbol filters %s: tick_size=%s step_size=%s min_qty=%s min_notional=%s",
            cfg.symbol,
            filt.tick_size,
            filt.step_size,
            filt.min_qty,
            filt.min_notional,
        )

        # B2: open orders
        open_orders = exchange.get_open_orders(cfg.symbol)
        log.info("Open orders: %d", len(open_orders))
        for o in open_orders[:20]:
            log.info("  open_order: %s", o)

        # B4: user data stream
        if cfg.b4_enabled:
            try:
                user_stream = BinanceUserDataStream(
                    exchange=exchange,
                    logger=us_log,
                    cfg=UserStreamConfig(
                        keepalive_interval_sec=cfg.user_keepalive_sec,
                        reconnect_backoff_sec=cfg.user_reconnect_backoff_sec,
                    ),
                    on_execution_report=lambda e: exec_reports.put(e),
                )
                user_stream.start()
                us_log.info("B4: user data stream started.")
            except Exception as e:
                us_log.warning("B4: failed to start user stream: %s", e)
                user_stream = None
        else:
            us_log.info("B4 disabled by config.")

        # B3: tiny market order (optional)
        if cfg.b3_enabled:
            try:
                log.info("B3: placing tiny MARKET order: %s %s qty=%s", cfg.b3_side, cfg.symbol, cfg.b3_qty)
                ack = exchange.place_market_order(
                    symbol=cfg.symbol,
                    side=cfg.b3_side,
                    quantity=cfg.b3_qty,
                    filters=filt,
                )
                log.info("B3: order ack: %s", ack)

                last_order_id: Optional[int] = None
                if isinstance(ack, dict):
                    oid = ack.get("orderId")
                    if isinstance(oid, int):
                        last_order_id = oid
                    elif isinstance(oid, str) and oid.isdigit():
                        last_order_id = int(oid)

                if user_stream is None:
                    log.info("B3: user stream not running => cannot wait for fills via executionReport.")
                else:
                    log.info("B3: waiting up to %ss for execution reports...", cfg.b3_wait_fills_sec)
                    deadline = time.time() + cfg.b3_wait_fills_sec
                    filled = False

                    while time.time() < deadline and not filled:
                        try:
                            evt = exec_reports.get(timeout=1.0)
                        except Empty:
                            continue

                        evt_oid = evt.get("i")
                        status = evt.get("X")

                        if last_order_id is not None and evt_oid != last_order_id:
                            continue

                        log.info("B3: executionReport matched orderId=%s status=%s", evt_oid, status)
                        if status == "FILLED":
                            filled = True

                    if not filled:
                        log.info("B3: did not observe FILLED within wait window (could still fill later).")

            except Exception as e:
                log.warning("B3: failed to place market order: %s", e)

        # B5: LIMIT order lifecycle test (place -> verify open -> cancel -> verify)
        if cfg.b5_enabled:
            try:
                if filt is None:
                    filt = exchange.get_symbol_filters(cfg.symbol)

                side_u = cfg.b5_side.upper()
                if side_u not in ("BUY", "SELL"):
                    raise ValueError("b5_side must be BUY or SELL")

                last = exchange.get_last_price(cfg.symbol)
                if side_u == "BUY":
                    target_price = last * (1.0 - cfg.b5_price_offset_pct)
                else:
                    target_price = last * (1.0 + cfg.b5_price_offset_pct)

                client_oid = f"B5-{int(time.time())}"
                log.info(
                    "B5: placing LIMIT %s %s qty=%s offset_pct=%.4f last=%.8f target=%.8f clientOid=%s",
                    side_u,
                    cfg.symbol,
                    cfg.b5_qty,
                    cfg.b5_price_offset_pct,
                    last,
                    target_price,
                    client_oid,
                )

                ack = exchange.place_limit_order(
                    symbol=cfg.symbol,
                    side=side_u,
                    quantity=cfg.b5_qty,
                    price=target_price,
                    client_order_id=client_oid,
                    filters=filt,
                )
                log.info("B5: LIMIT order ack: %s", ack)

                order_id: Optional[int] = None
                if isinstance(ack, dict):
                    oid = ack.get("orderId")
                    if isinstance(oid, int):
                        order_id = oid
                    elif isinstance(oid, str) and oid.isdigit():
                        order_id = int(oid)

                oo1 = exchange.get_open_orders(cfg.symbol)
                log.info("B5: open orders after placement: %d", len(oo1))

                log.info("B5: waiting %ss then canceling.", cfg.b5_cancel_after_sec)
                time.sleep(max(0, cfg.b5_cancel_after_sec))

                exchange.cancel_order(
                    symbol=cfg.symbol,
                    order_id=order_id,
                    client_order_id=client_oid,
                )
                log.info("B5: cancel requested (order_id=%s clientOid=%s).", order_id, client_oid)

                oo2 = exchange.get_open_orders(cfg.symbol)
                log.info("B5: open orders after cancel: %d", len(oo2))

                # If user stream is running, we’ll opportunistically print a few exec reports after B5
                if user_stream is not None:
                    end = time.time() + 10.0
                    while time.time() < end:
                        try:
                            evt = exec_reports.get(timeout=1.0)
                        except Empty:
                            continue
                        # executionReport has i=orderId, X=status
                        log.info("B5: executionReport: orderId=%s status=%s", evt.get("i"), evt.get("X"))

            except Exception as e:
                log.warning("B5: limit lifecycle test failed: %s", e)

    else:
        log.warning(
            "Private endpoints disabled (missing creds). "
            "If this is PAPER mode, make sure BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET exist in env."
        )

    # -----------------------------------------------------------------------
    # Market data stream (B1): BinanceKlineStream
    # -----------------------------------------------------------------------
    use_testnet_ws = cfg.mode == RunMode.PAPER
    kline_stream = BinanceKlineStream(
        symbol=cfg.symbol,
        interval=cfg.interval,
        use_testnet=use_testnet_ws,
        logger=md_log,
    )
    kline_stream.start()

    log.info(
        "Market stream started (symbol=%s interval=%s use_testnet_ws=%s). Waiting for CLOSED candles...",
        cfg.symbol,
        cfg.interval,
        use_testnet_ws,
    )

    try:
        while True:
            try:
                candle = kline_stream.next_closed_candle(timeout_sec=70.0)  # 1m candles => wait >60s
                log.info(
                    "CLOSED candle %s O=%.8f H=%.8f L=%.8f C=%.8f V=%.8f",
                    candle.timestamp.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                )
            except TimeoutError:
                st2 = kline_stream.status()
                log.debug(
                    "No CLOSED candle yet. stream_connected=%s last_msg=%s url=%s",
                    st2.connected,
                    st2.last_message_ts_utc.isoformat() if st2.last_message_ts_utc else None,
                    st2.url,
                )
                continue
    except KeyboardInterrupt:
        log.info("Stopping live runtime (KeyboardInterrupt)...")
    finally:
        try:
            kline_stream.stop()
        except Exception:
            pass
        if user_stream is not None:
            try:
                user_stream.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
