# app/live_trade.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional, List, Dict, Tuple

try:
    import yaml  # type: ignore
except ImportError as e:
    raise ImportError("Missing dependency: PyYAML. Install via: pip install pyyaml") from e

from infra.logging_setup import init_logging, get_logger
from infra.config.engine_config import RunMode
from infra.secrets import EnvSecretsProvider

from infra.exchange.binance_spot import BinanceSpotExchange, SymbolFilters
from infra.marketdata.binance_kline_stream import BinanceKlineStream
from infra.marketdata.binance_user_stream import BinanceUserDataStream, UserStreamConfig

from core.engine_actions import EngineAction, EngineActionType  # PLACE_ORDER etc. :contentReference[oaicite:1]{index=1}
from core.models import Candle, Side, OrderType, OrderFilledEvent

from core.live.order_manager import OrderManager, decode_grid_tag_from_client_order_id

from core.results.live_repository import LiveRepository
from core.live.equity_tracker import EquityTracker

from core.strategy.base import BaseStrategy
from core.strategy.grid_strategy_simple import (
    GridConfig,
    GridRangeMode,
    GridSpacing,
    SimpleGridStrategy,
)
from core.strategy.grid_strategy_dynamic import (
    DynamicGridConfig,
    DynamicGridStrategy,
)

# Config models used elsewhere in your codebase (simple_grid_backtest imports these). :contentReference[oaicite:2]{index=2}
from infra.config import GridStrategyConfig
from infra.config.strategy_grid import DynamicGridStrategyConfig


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_ts() -> datetime:
    return datetime.now(timezone.utc)


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


def _split_symbol(symbol: str) -> Tuple[str, str]:
    """
    Best-effort split for spot symbols. Works for common quotes like USDT, BUSD, USDC, EUR, BTC, ETH.
    """
    s = symbol.upper()
    for q in ("USDT", "USDC", "BUSD", "FDUSD", "EUR", "BTC", "ETH", "BNB", "TRY"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)], q
    # fallback: treat last 4 as quote (not perfect)
    return s[:-4], s[-4:]


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _ms_to_dt_utc(ms: Any) -> datetime:
    ms_i = int(ms)
    return datetime.fromtimestamp(ms_i / 1000.0, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Safety config (B9)
# ---------------------------------------------------------------------------

@dataclass
class LiveSafetyConfig:
    max_open_orders: int = 20
    max_notional_per_order: float = 50.0
    max_position_base: float = 10_000.0
    cancel_all_on_shutdown: bool = True


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
    b5_price_offset_pct: float = 0.01
    b5_cancel_after_sec: int = 5

    # B6
    b6_enabled: bool = True

    # B8 shadow mode
    b8_shadow_enabled: bool = False

    # B9 execute mode (config-based, no env vars)
    execute_enabled: bool = False

    cancel_open_orders_on_startup: bool = False
    cancel_open_orders_only_managed: bool = True
    cancel_open_orders_prefixes: Optional[List[str]] = None

    safety: LiveSafetyConfig = field(default_factory=LiveSafetyConfig)

    # logging hygiene
    balances_log_assets: Optional[List[str]] = None

    # strategy section (raw dict) to avoid duplicating too many models here
    strategy_raw: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str) -> Tuple[LiveTradeConfig, Dict[str, Any]]:
    """
    Load YAML and return:
      - parsed LiveTradeConfig
      - raw dict (saved in session.json)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    name = raw.get("name") or raw.get("run_name") or "xrp_live_phase_b"
    engine = raw.get("engine") if isinstance(raw.get("engine"), dict) else {}
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    logging = raw.get("logging") if isinstance(raw.get("logging"), dict) else {}
    live = raw.get("live") if isinstance(raw.get("live"), dict) else {}
    strat = raw.get("strategy") if isinstance(raw.get("strategy"), dict) else {}

    mode_raw = engine.get("mode") or raw.get("mode") or "paper"
    mode = _parse_mode(str(mode_raw))

    symbol = data.get("symbol") or raw.get("symbol") or "XRPUSDT"
    interval = data.get("timeframe") or raw.get("interval") or "1m"

    log_level = str(logging.get("level") or raw.get("log_level") or "INFO")

    require_private = bool(live.get("require_private_endpoints", raw.get("require_private_endpoints", True)))

    # B3
    b3_legacy = raw.get("b3_test_order") if isinstance(raw.get("b3_test_order"), dict) else {}
    b3_enabled = bool(live.get("b3_enabled", b3_legacy.get("enabled", False)))
    b3_side = str(live.get("b3_side", b3_legacy.get("side", "BUY"))).upper()
    b3_qty = float(live.get("b3_qty", b3_legacy.get("qty", 0.1)))
    b3_wait = int(live.get("b3_wait_fills_sec", b3_legacy.get("wait_fills_sec", 20)))

    # B4
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

    # B6/B8
    b6_enabled = bool(live.get("b6_enabled", True))
    b8_shadow_enabled = bool(live.get("b8_shadow_enabled", False))

    cancel_on_startup = bool(live.get("cancel_open_orders_on_startup", False))
    cancel_only_managed = bool(live.get("cancel_open_orders_only_managed", True))
    cancel_prefixes = live.get("cancel_open_orders_prefixes", None)
    if cancel_prefixes is not None and not isinstance(cancel_prefixes, list):
        cancel_prefixes = None

    # B9 execute
    execute_enabled = bool(live.get("execute_enabled", False))
    safety_raw = live.get("safety", {}) if isinstance(live.get("safety", {}), dict) else {}
    safety = LiveSafetyConfig(
        max_open_orders=int(safety_raw.get("max_open_orders", 20)),
        max_notional_per_order=float(safety_raw.get("max_notional_per_order", 50.0)),
        max_position_base=float(safety_raw.get("max_position_base", 10_000.0)),
        cancel_all_on_shutdown=bool(safety_raw.get("cancel_all_on_shutdown", True)),
    )

    balances_log_assets = live.get("balances_log_assets", None)
    if balances_log_assets is not None and not isinstance(balances_log_assets, list):
        balances_log_assets = None

    cfg = LiveTradeConfig(
        run_name=str(name),
        mode=mode,
        config_path=path,
        log_level=log_level,
        symbol=str(symbol),
        interval=str(interval),
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
        b6_enabled=b6_enabled,
        b8_shadow_enabled=b8_shadow_enabled,
        execute_enabled=execute_enabled,
        cancel_open_orders_on_startup=cancel_on_startup,
        cancel_open_orders_only_managed=cancel_only_managed,
        cancel_open_orders_prefixes=cancel_prefixes,
        safety=safety,
        
        balances_log_assets=balances_log_assets,
        strategy_raw=strat,
    )
    return cfg, raw


# ---------------------------------------------------------------------------
# Strategy builder (shared with backtests)
# ---------------------------------------------------------------------------

def _parse_range_mode(s: str) -> GridRangeMode:
    v = (s or "").strip().lower()
    if v in {"percent", "pct"}:
        return GridRangeMode.PERCENT
    if v in {"absolute", "abs"}:
        return GridRangeMode.ABSOLUTE
    return GridRangeMode.PERCENT


def _parse_spacing(s: str) -> GridSpacing:
    v = (s or "").strip().lower()
    if v in {"geometric", "geo"}:
        return GridSpacing.GEOMETRIC
    return GridSpacing.ARITHMETIC


def _build_strategy_from_raw(strategy_raw: Dict[str, Any]) -> BaseStrategy:
    if not strategy_raw:
        raise ValueError("Missing strategy section in config.")

    kind = str(strategy_raw.get("kind", "")).strip().lower()

    if kind == "grid.dynamic":
        strat_cfg = DynamicGridStrategyConfig.parse_obj(strategy_raw)
        cfg = DynamicGridConfig(
            symbol=strat_cfg.symbol,
            base_order_size=float(strat_cfg.base_order_size),
            n_levels=int(strat_cfg.n_levels),
            range_mode=_parse_range_mode(str(strat_cfg.range_mode)),
            spacing=_parse_spacing(str(strat_cfg.spacing)),
            lower_pct=float(strat_cfg.lower_pct) if strat_cfg.lower_pct is not None else None,
            upper_pct=float(strat_cfg.upper_pct) if strat_cfg.upper_pct is not None else None,
            lower_price=float(strat_cfg.lower_price) if strat_cfg.lower_price is not None else None,
            upper_price=float(strat_cfg.upper_price) if strat_cfg.upper_price is not None else None,
            range_atr_period=int(strat_cfg.range_atr_period) if getattr(strat_cfg, "range_atr_period", None) else None,
            #recenter_policy=getattr(strat_cfg, "recenter_policy", None),
        )
        return DynamicGridStrategy(cfg)

    if kind == "grid.simple":
        strat_cfg = GridStrategyConfig.parse_obj(strategy_raw)
        cfg = GridConfig(
            symbol=strat_cfg.symbol,
            base_order_size=float(strat_cfg.base_order_size),
            n_levels=int(strat_cfg.n_levels),
            range_mode=_parse_range_mode(str(strat_cfg.range_mode)),
            spacing=_parse_spacing(str(strat_cfg.spacing)),
            lower_pct=float(strat_cfg.lower_pct) if strat_cfg.lower_pct is not None else None,
            upper_pct=float(strat_cfg.upper_pct) if strat_cfg.upper_pct is not None else None,
            lower_price=float(strat_cfg.lower_price) if strat_cfg.lower_price is not None else None,
            upper_price=float(strat_cfg.upper_price) if strat_cfg.upper_price is not None else None,
        )
        return SimpleGridStrategy(cfg)

    raise ValueError(f"Unknown strategy.kind={kind!r}")


# ---------------------------------------------------------------------------
# Execute-mode safety checks (B9)
# ---------------------------------------------------------------------------

def _count_managed_open_orders(order_mgr: OrderManager) -> int:
    for attr in ("open_orders", "_open_orders", "_orders"):
        if hasattr(order_mgr, attr):
            try:
                v = getattr(order_mgr, attr)
                return len(v)
            except Exception:
                pass
    if hasattr(order_mgr, "count_open"):
        try:
            return int(order_mgr.count_open())
        except Exception:
            pass
    return 0


def _cancel_all_managed(order_mgr: OrderManager, symbol: str, reason: str) -> None:
    for meth in ("cancel_all", "cancel_all_open", "cancel_all_open_orders", "cancel_all_managed"):
        if hasattr(order_mgr, meth):
            getattr(order_mgr, meth)(symbol=symbol, reason=reason)  # type: ignore
            return
    # fallback: no-op if not available
    raise RuntimeError("OrderManager has no cancel_all_* method; implement one for kill-switch safety.")


def _exec_place_order(
    *,
    action: EngineAction,
    candle: Candle,
    exchange: BinanceSpotExchange,
    order_mgr: OrderManager,
    filters: SymbolFilters,
    safety: LiveSafetyConfig,
    base_asset: str,
    quote_asset: str,
    log,
    repo: LiveRepository,
) -> None:
    order = action.order
    if order is None:
        return

    # Estimate notional for MARKET using candle.close
    px = float(order.price or 0.0)
    if order.type == OrderType.MARKET:
        px = float(candle.close)

    qty = float(order.qty or 0.0)
    if qty <= 0:
        repo.append_event("b9.refused", {"reason": "qty<=0", "client_tag": order.client_tag})
        return

    notional = px * qty
    if safety.max_notional_per_order > 0 and notional > float(safety.max_notional_per_order):
        repo.append_event(
            "b9.refused",
            {"reason": "max_notional_per_order", "notional": notional, "limit": safety.max_notional_per_order, "client_tag": order.client_tag},
        )
        return

    # Open order cap
    n_open = _count_managed_open_orders(order_mgr)
    if safety.max_open_orders > 0 and n_open >= int(safety.max_open_orders):
        repo.append_event(
            "b9.refused",
            {"reason": "max_open_orders", "open_orders": n_open, "limit": safety.max_open_orders, "client_tag": order.client_tag},
        )
        return

    # Position cap (spot base holding cap)
    balances = exchange.get_balances(non_zero_only=False)
    base_free, base_locked = balances.get(base_asset, (0.0, 0.0))
    base_total = float(base_free) + float(base_locked)

    # If BUY, this increases base exposure; if SELL, it decreases (so allow)
    if order.side == Side.BUY and safety.max_position_base > 0 and (base_total + qty) > float(safety.max_position_base):
        repo.append_event(
            "b9.refused",
            {"reason": "max_position_base", "base_total": base_total, "buy_qty": qty, "limit": safety.max_position_base, "client_tag": order.client_tag},
        )
        return

    side_u = order.side.value.upper()

    # Submit through OrderManager (the whole point of B6/B9)
    if order.type == OrderType.MARKET:
        res = order_mgr.place_market(side=side_u, qty=qty, intent="grid", tag=order.client_tag)
        repo.append_event("b9.submitted", {"type": "MARKET", "side": side_u, "qty": qty, "client_tag": order.client_tag, "result": str(res)})
        log.info("B9: submitted MARKET %s qty=%.8f tag=%s", side_u, qty, order.client_tag)

    elif order.type == OrderType.LIMIT:
        if px <= 0:
            repo.append_event("b9.refused", {"reason": "limit_px<=0", "client_tag": order.client_tag})
            return
        res = order_mgr.place_limit(side=side_u, qty=qty, price=px, intent="grid", tag=order.client_tag)
        repo.append_event("b9.submitted", {"type": "LIMIT", "side": side_u, "qty": qty, "price": px, "client_tag": order.client_tag, "result": str(res)})
        log.info("B9: submitted LIMIT %s qty=%.8f price=%.8f tag=%s", side_u, qty, px, order.client_tag)

    else:
        repo.append_event("b9.refused", {"reason": f"unsupported_order_type:{order.type}", "client_tag": order.client_tag})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/test_live_trade.yml")
    args = parser.parse_args()

    cfg, raw_cfg = load_config(args.config)

    log_file = init_logging(run_name=cfg.run_name, level_name=cfg.log_level)
    log = get_logger("live")
    ex_log = get_logger("live.exchange")
    md_log = get_logger("live.marketdata")
    us_log = get_logger("live.userstream")
    ord_log = get_logger("live.orders")

    log.info("=== Starting LiveTrading runtime (B1..B9 harness) ===")
    log.info("Run: %s", cfg.run_name)
    log.info("Mode: %s", cfg.mode.value)
    log.info("Config: %s", cfg.config_path)
    log.info("Log level: %s", cfg.log_level)
    log.info("Log file: %s", str(log_file))
    log.info("Symbol: %s | Interval: %s", cfg.symbol, cfg.interval)
    log.info("B8 shadow: %s | B9 execute: %s", cfg.b8_shadow_enabled, cfg.execute_enabled)

    base_asset, quote_asset = _split_symbol(cfg.symbol)

    # -----------------------------------------------------------------------
    # LiveRepository (session.json + events.jsonl)
    # -----------------------------------------------------------------------
    base_dir = "runs/live"
    repo = LiveRepository(base_dir=base_dir, run_name=cfg.run_name)
    run_id = repo.paths.root.name
    repo.write_session(
        {
            "run_id": run_id,
            "run_name": cfg.run_name,
            "mode": cfg.mode.value,
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "started_at": _utcnow(),
            "config_path": cfg.config_path,
            "config": raw_cfg,
        }
    )

    # -----------------------------------------------------------------------
    # Secrets: ALWAYS via EnvSecretsProvider (keys), but safeties are config-based.
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
        log.warning("Secrets not available (%s). Continuing PUBLIC-ONLY mode.", e)

    has_private_creds = bool(api_key) and bool(api_secret)

    # -----------------------------------------------------------------------
    # Exchange bootstrap (B1)
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
    # Strategy (B8/B9)
    # -----------------------------------------------------------------------
    strategy: Optional[BaseStrategy] = None
    if cfg.b8_shadow_enabled or cfg.execute_enabled:
        strategy = _build_strategy_from_raw(cfg.strategy_raw)
        repo.append_event("b8.strategy_loaded", {"kind": cfg.strategy_raw.get("kind"), "symbol": cfg.symbol})
        log.info("Strategy loaded: %s", cfg.strategy_raw.get("kind"))

    # -----------------------------------------------------------------------
    # Private endpoints (B2/B4/B3/B5/B6/B9) only when creds exist
    # -----------------------------------------------------------------------
    exec_reports: "Queue[dict[str, Any]]" = Queue()
    user_stream: Optional[BinanceUserDataStream] = None
    order_mgr: Optional[OrderManager] = None
    filters: Optional[SymbolFilters] = None
    equity: Optional[EquityTracker] = None

    if has_private_creds:
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

        filters = exchange.get_symbol_filters(cfg.symbol)
        log.info(
            "Symbol filters %s: tick_size=%s step_size=%s min_qty=%s min_notional=%s",
            cfg.symbol,
            filters.tick_size,
            filters.step_size,
            filters.min_qty,
            filters.min_notional,
        )

        open_orders = exchange.get_open_orders(cfg.symbol)
        log.info("Open orders: %d", len(open_orders))

        # --- Startup cleanup config
        prefixes = tuple(getattr(cfg, "cancel_open_orders_prefixes", None) or ["GT-", "LT-", "LT_"])
        do_cleanup = bool(getattr(cfg, "cancel_open_orders_on_startup", False)) and bool(open_orders)
        only_managed = bool(getattr(cfg, "cancel_open_orders_only_managed", True))

        # --- Safety guard: never allow "cancel ALL" on LIVE unless explicitly allowed (we refuse by default)
        if cfg.mode == RunMode.LIVE and do_cleanup and not only_managed:
            raise RuntimeError(
                "Refusing to cancel ALL open orders on LIVE. Set cancel_open_orders_only_managed=true."
            )

        # --- Create OrderManager early so cleanup + reconciliation share one code path
        order_mgr = None
        if cfg.b6_enabled or do_cleanup:
            order_mgr = OrderManager(
                run_name=cfg.run_name,
                symbol=cfg.symbol,
                exchange=exchange,
                filters=filters,
                logger=ord_log,
            )
            # initial reconcile (so it knows exchange ids, client ids, etc.)
            order_mgr.reconcile_open_orders(open_orders)

        # --- Startup cleanup using OrderManager (preferred)
        if do_cleanup:
            if order_mgr is None:
                # Fallback (shouldn't happen if you keep b6_enabled=true)
                log.warning("Startup cleanup requested but OrderManager is not available; falling back to direct REST cancels.")
                to_cancel = (
                    [o for o in open_orders if str(o.get("clientOrderId", "")).startswith(prefixes)]
                    if only_managed
                    else list(open_orders)
                )
                attempted = 0
                for o in to_cancel:
                    attempted += 1
                    oid = o.get("orderId")
                    try:
                        exchange.cancel_order(symbol=cfg.symbol, order_id=oid)
                    except Exception as e:
                        log.warning("Failed to cancel orderId=%s: %s", oid, e)
            else:
                log.warning(
                    "Startup cleanup: canceling open orders (only_managed=%s prefixes=%s)",
                    only_managed,
                    prefixes,
                )
                if only_managed:
                    attempted = order_mgr.cancel_all_managed(reason="startup_cleanup", prefixes=prefixes)
                else:
                    attempted = order_mgr.cancel_all_open_orders(reason="startup_cleanup_all")

            # Re-fetch truth + re-reconcile
            time.sleep(0.2)  # tiny delay helps the exchange reflect cancels
            open_orders = exchange.get_open_orders(cfg.symbol)
            log.info("Open orders after startup cleanup: %d", len(open_orders))
            if order_mgr is not None:
                order_mgr.reconcile_open_orders(open_orders)

            repo.append_event(
                "startup.cleanup",
                {
                    "attempted": int(attempted),
                    "only_managed": bool(only_managed),
                    "prefixes": list(prefixes),
                    "open_orders_after": int(len(open_orders)),
                },
            )

        # --- Final reconcile event (single source of truth)
        if cfg.b6_enabled and order_mgr is not None:
            repo.append_event("b6.reconciled", {"open_orders": len(open_orders)})


        equity = EquityTracker(exchange=exchange, symbol=cfg.symbol)
        #equity.refresh()
        px, b, q = equity.snapshot()
        repo.append_event("equity.snapshot", equity.snapshot_json() if hasattr(equity, "snapshot_json") else {"price": px})
        repo.append_equity(
            symbol=cfg.symbol,
            price=float(px),
            base_free=float(getattr(b, "free", 0.0)),
            base_locked=float(getattr(b, "locked", 0.0)),
            quote_free=float(getattr(q, "free", 0.0)),
            quote_locked=float(getattr(q, "locked", 0.0)),
        )

        # B4 user stream first (so we don't miss reports)
        if cfg.b4_enabled:
            def _on_exec_report(evt: Dict[str, Any]) -> None:
                repo.append_event("b4.execution_report", evt)
                if order_mgr is not None:
                    order_mgr.on_execution_report(evt)
                exec_reports.put(evt)
                # --- B7 persistence: executionReport -> orders.csv / fills.csv
                try:
                    symbol = str(evt.get("s") or cfg.symbol)
                    side = str(evt.get("S") or "")
                    otype = str(evt.get("o") or "")
                    status = str(evt.get("X") or "")
                    exec_type = str(evt.get("x") or "")
                    exchange_oid = str(evt.get("i") or "")

                    # Prefer 'C' (origClientOrderId) when present (cancel reports), else 'c'
                    client_oid = str(evt.get("C") or evt.get("c") or "")

                    qty = float(evt.get("q") or 0.0)
                    price = float(evt.get("p") or 0.0)

                    if hasattr(repo, "append_order"):
                        repo.append_order(
                            symbol=symbol,
                            side=side,
                            order_type=otype,
                            client_order_id=client_oid,
                            exchange_order_id=exchange_oid,
                            status=status,
                            qty=qty,
                            price=price,
                        )

                    # Fills come as exec_type=TRADE with last fill qty/price
                    if exec_type == "TRADE":
                        fill_qty = float(evt.get("l") or 0.0)
                        fill_price = float(evt.get("L") or 0.0)
                        cum_qty = float(evt.get("z") or 0.0)
                        cum_quote = float(evt.get("Z") or 0.0)
                        if fill_qty > 0 and fill_price > 0 and hasattr(repo, "append_fill"):
                            repo.append_fill(
                                symbol=symbol,
                                side=side,
                                order_type=otype,
                                client_order_id=client_oid,
                                exchange_order_id=exchange_oid,
                                fill_qty=fill_qty,
                                fill_price=fill_price,
                                cum_qty=cum_qty,
                                cum_quote=cum_quote,
                            )
                except Exception as _e:
                    # never break the stream due to persistence
                    pass

                # Feed strategy fills (if any) to keep it stateful; SimpleGrid uses client_tag mapping :contentReference[oaicite:3]{index=3}
                if strategy is not None:
                    status = str(evt.get("X", "")).upper()
                    exec_type = str(evt.get("x", "")).upper()
                    if exec_type == "TRADE" and status in {"PARTIALLY_FILLED", "FILLED"}:
                        last_qty = _to_float(evt.get("l"))
                        last_px = _to_float(evt.get("L"))
                        if last_qty > 0 and last_px > 0:
                            filled_at = _ms_to_dt_utc(evt.get("T", evt.get("E", int(time.time() * 1000))))
                            raw_client_oid = str(evt.get("C") or evt.get("c") or "")
                            decoded_tag = decode_grid_tag_from_client_order_id(raw_client_oid)
                            tag = decoded_tag or raw_client_oid

                            side = Side.BUY if str(evt.get("S", "")).upper() == "BUY" else Side.SELL
                            ofe = OrderFilledEvent(
                                order_id=str(evt.get("i")),
                                symbol=str(evt.get("s") or cfg.symbol),
                                side=side,
                                price=float(last_px),
                                qty=float(last_qty),
                                filled_at=filled_at,
                                client_tag=tag,
                            )
                            strategy.on_order_filled(ofe)
                            repo.append_event("strategy.on_order_filled", {"order_id": ofe.order_id, "tag": tag, "qty": ofe.qty, "price": ofe.price})

            user_stream = BinanceUserDataStream(
                exchange=exchange,
                logger=us_log,
                cfg=UserStreamConfig(
                    keepalive_interval_sec=cfg.user_keepalive_sec,
                    reconnect_backoff_sec=cfg.user_reconnect_backoff_sec,
                ),
                on_execution_report=_on_exec_report,
            )
            user_stream.start()

            if not user_stream.wait_until_connected(timeout_sec=10.0):
                us_log.warning("User stream did not connect within timeout; disabling B3/B5/B9 for safety.")
                cfg.b3_enabled = False
                cfg.b5_enabled = False
                cfg.execute_enabled = False
            else:
                us_log.info("B4: user data stream started.")
                repo.append_event("b4.started", {"ok": True})
        else:
            us_log.info("B4 disabled by config.")
            repo.append_event("b4.started", {"ok": False, "reason": "disabled"})

        # B3 + B5 MUST go through OrderManager (fix unknown-order race)
        if cfg.b3_enabled:
            if order_mgr is None or filters is None:
                log.warning("B3 enabled but OrderManager not available (b6_enabled=false?). Skipping.")
            else:
                try:
                    tag = f"GT-{int(time.time())}"
                    repo.append_event("b3.submitted", {"client_order_id": tag, "side": cfg.b3_side, "qty": cfg.b3_qty})

                    if order_mgr is None:
                        raise RuntimeError("B3 enabled but OrderManager is not initialized (b6_enabled must be true).")

                    log.info("B3: placing tiny MARKET order via OrderManager: %s %s qty=%s", cfg.b3_side, cfg.symbol, cfg.b3_qty)
                    mo = order_mgr.place_market(
                        side=cfg.b3_side,
                        qty=cfg.b3_qty,
                        intent="b3",
                        tag="B3",
                    )
                    repo.append_event("b3.submitted", {"client_order_id": mo.client_order_id, "side": mo.side, "qty": mo.qty})

                    log.info("B3: waiting up to %ss for FILLED...", cfg.b3_wait_fills_sec)
                    deadline = time.time() + cfg.b3_wait_fills_sec
                    while time.time() < deadline:
                        if mo.status.upper() == "FILLED":
                            log.info("B3: FILLED clientId=%s oid=%s cumQty=%.8f cumQuote=%.8f",
                                    mo.client_order_id, mo.exchange_order_id, mo.cum_qty, mo.cum_quote)
                            repo.append_event("b3.filled", {"client_order_id": mo.client_order_id, "order_id": mo.exchange_order_id, "cum_qty": mo.cum_qty, "cum_quote": mo.cum_quote})
                            break
                        if mo.is_terminal():
                            log.info("B3: terminal but not FILLED: status=%s clientId=%s", mo.status, mo.client_order_id)
                            break
                        time.sleep(0.1)
                    else:
                        log.info("B3: did not observe FILLED within wait window (could still fill later).")

                except Exception as e:
                    log.warning("B3 failed: %s", e)

        if cfg.b5_enabled:
            if order_mgr is None or filters is None:
                log.warning("B5 enabled but OrderManager not available (b6_enabled=false?). Skipping.")
                repo.append_event("b5.skipped", {"reason": "order_manager_unavailable", "b6_enabled": cfg.b6_enabled},)
            else:
                try:
                    side_u = cfg.b5_side.upper()
                    last = exchange.get_last_price(cfg.symbol)
                    target_price = last * (1.0 - cfg.b5_price_offset_pct) if side_u == "BUY" else last * (1.0 + cfg.b5_price_offset_pct)
                    client_tag = f"GT-{int(time.time())}"
                    repo.append_event("b5.submitted", {"client_order_id": client_tag, "side": side_u, "qty": cfg.b5_qty, "price": target_price})

                    log.info(
                        "B5: placing LIMIT via OrderManager: %s %s qty=%s target=%.8f tag=%s",
                        side_u, cfg.symbol, cfg.b5_qty, target_price, client_tag,
                    )
                    lo = order_mgr.place_limit(side=side_u, qty=cfg.b5_qty, price=target_price, intent="b5", tag=client_tag)

                    log.info("B5: waiting %ss then canceling.", cfg.b5_cancel_after_sec)
                    time.sleep(max(0, cfg.b5_cancel_after_sec))

                    repo.append_event("b5.cancel_requested", {"client_order_id": lo.client_order_id})
                    order_mgr.cancel(client_order_id=lo.client_order_id)
                    repo.append_event("b5.canceled", {"client_order_id": lo.client_order_id})
                    log.info("B5: cancel requested for client_order_id=%s", lo.client_order_id)

                except Exception as e:
                    log.warning("B5 failed: %s", e)

    else:
        log.warning("Private endpoints disabled (missing creds).")
        repo.append_event("private.disabled", {"reason": "missing_creds"})

    # -----------------------------------------------------------------------
    # Market data stream (B1)
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
                candle = kline_stream.next_closed_candle(timeout_sec=70.0)
            except TimeoutError:
                st2 = kline_stream.status()
                log.debug(
                    "No CLOSED candle yet. stream_connected=%s last_msg=%s url=%s",
                    st2.connected,
                    st2.last_message_ts_utc.isoformat() if st2.last_message_ts_utc else None,
                    st2.url,
                )
                continue

            # B1 log + persist candle
            log.info(
                "CLOSED candle %s O=%.8f H=%.8f L=%.8f C=%.8f V=%.8f",
                candle.timestamp.isoformat(),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )
            repo.append_event(
                "b1.closed_candle",
                {
                    "ts": candle.timestamp.isoformat(),
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                },
            )

            # B8/B9 need strategy + private pieces
            if strategy is None or not (cfg.b8_shadow_enabled or cfg.execute_enabled):
                continue

            if equity is not None:
                equity.refresh(last_price=float(candle.close))
                px, b, q = equity.snapshot()
                repo.append_event("equity.snapshot", equity.snapshot_json() if hasattr(equity, "snapshot_json") else {"price": px})
                repo.append_equity(
                    symbol=cfg.symbol,
                    price=float(px),
                    base_free=float(getattr(b, "free", 0.0)),
                    base_locked=float(getattr(b, "locked", 0.0)),
                    quote_free=float(getattr(q, "free", 0.0)),
                    quote_locked=float(getattr(q, "locked", 0.0)),
                )
            # Build AccountState for strategy
            account_state = equity.account_state() if equity is not None else None
            if account_state is None:
                repo.append_event("b8.skipped", {"reason": "no_account_state"})
                continue

            actions = strategy.on_candle(candle, account_state) or []
            repo.append_event(
                "b8.strategy_actions",
                {
                    "n": len(actions),
                    "actions": [str(a) for a in actions],
                },
            )

            # B8 shadow only: do not execute
            if not cfg.execute_enabled:
                continue

            # B9 execute-mode: only PLACE_ORDER actions, via OrderManager
            if not has_private_creds or order_mgr is None or filters is None:
                repo.append_event("b9.blocked", {"reason": "missing_private_or_order_mgr"})
                continue

            for action in actions:
                if action.type != EngineActionType.PLACE_ORDER:
                    repo.append_event("b9.ignored", {"type": str(action.type), "reason": "execute_only_place_order"})
                    continue

                _exec_place_order(
                    action=action,
                    candle=candle,
                    exchange=exchange,
                    order_mgr=order_mgr,
                    filters=filters,
                    safety=cfg.safety,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    log=log,
                    repo=repo,
                )

    except KeyboardInterrupt:
        log.info("Stopping live runtime (KeyboardInterrupt)...")
        repo.append_event("runtime.stop", {"reason": "KeyboardInterrupt"})
    finally:
        # Kill-switch: cancel all managed open orders
        if cfg.safety.cancel_all_on_shutdown and order_mgr is not None and has_private_creds:
            try:
                log.warning("Kill-switch: canceling all managed open orders on shutdown...")
                _cancel_all_managed(order_mgr, symbol=cfg.symbol, reason="shutdown_killswitch")
                repo.append_event("killswitch.cancel_all", {"ok": True})
            except Exception as e:
                repo.append_event("killswitch.cancel_all", {"ok": False, "error": str(e)})

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
