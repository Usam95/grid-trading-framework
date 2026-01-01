# app/trading/runtime.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any, Dict, Optional

from infra.logging_setup import init_logging, get_logger
from infra.secrets import EnvSecretsProvider
from infra.config.engine_config import RunMode

from infra.exchange.binance_spot import BinanceSpotExchange, SymbolFilters
from infra.marketdata.binance_kline_stream import BinanceKlineStream

from core.results.live_repository import LiveRepository
from core.strategy.base import BaseStrategy
from core.models import OrderFilledEvent, Side
from core.live.order_manager import OrderManager, decode_grid_tag_from_client_order_id
from core.live.equity_tracker import EquityTracker
from typing import Optional
# add near other imports
from core.live.pnl_ledger import FillLedger, BuyHoldBaseline
from app.trading.config import TradingSettings, resolve_use_testnet_ws
from app.trading.utils import utcnow_iso, mask_secret, split_symbol, to_float, ms_to_dt_utc
from app.trading.user_stream import start_user_stream
from app.trading.account_state import build_account_state
from app.trading.execution import execute_actions


@dataclass
class TradingContext:
    settings: TradingSettings

    # derived
    symbol: str
    interval: str
    base_asset: str
    quote_asset: str

    # logging
    log: object
    ex_log: object
    md_log: object
    us_log: object
    ord_log: object

    # persistence
    repo: LiveRepository

    # exchange + runtime
    exchange: BinanceSpotExchange
    strategy: BaseStrategy

    api_key: str
    api_secret: str
    has_private_creds: bool

    filters: Optional[SymbolFilters] = None
    order_mgr: Optional[OrderManager] = None
    equity: Optional[EquityTracker] = None
    user_stream: Optional[object] = None
    kline_stream: Optional[object] = None

    # thread-safe queues (user stream thread -> main thread)
    exec_reports: Queue[Dict[str, Any]] = field(default_factory=Queue)
    fill_events: Queue[OrderFilledEvent] = field(default_factory=Queue)

    # inside TradingContext dataclass
    pnl_ledger: Optional[FillLedger] = None
    buyhold: Optional[BuyHoldBaseline] = None


    @property
    def trading(self):
        return self.settings.trading

def persist_pnl(ctx: TradingContext, *, reason: str, last_price: Optional[float] = None) -> None:
    if ctx.pnl_ledger is None or ctx.buyhold is None:
        return
    try:
        px = float(last_price) if (last_price is not None and float(last_price) > 0.0) else float(ctx.pnl_ledger.start_price)
        snap = ctx.pnl_ledger.snapshot(last_price=px)

        bh_eq = ctx.buyhold.equity(last_price=px)
        bh_ret = ctx.buyhold.return_pct(last_price=px)

        ctx.repo.append_pnl(
            symbol=ctx.symbol,
            price=float(px),
            cash_quote=float(snap["cash_quote"]),
            base_position=float(snap["base_position"]),
            invested_cost_quote=float(snap["invested_cost_quote"]),
            fees_paid_quote=float(snap["fees_paid_quote"]),
            realized_pnl_quote=float(snap["realized_pnl_quote"]),
            unrealized_pnl_quote=float(snap["unrealized_pnl_quote"]),
            total_pnl_quote=float(snap["total_pnl_quote"]),
            ledger_equity_quote=float(snap["ledger_equity_quote"]),
            buyhold_equity_quote=float(bh_eq),
            buyhold_return_pct=float(bh_ret),
            reason=str(reason),
        )
        ctx.repo.append_event("pnl.persisted", {"reason": reason, "price": float(px)})
    except Exception as e:
        ctx.repo.append_event("pnl.persist_failed", {"reason": reason, "error": str(e)})

def persist_equity(ctx: TradingContext, *, reason: str, last_price: Optional[float] = None) -> None:
    if ctx.equity is None:
        return
    try:
        px, b, q = ctx.equity.snapshot(last_price=last_price)
        ctx.repo.append_equity(
            symbol=ctx.symbol,
            price=float(px),
            base_free=float(getattr(b, "free", 0.0)),
            base_locked=float(getattr(b, "locked", 0.0)),
            quote_free=float(getattr(q, "free", 0.0)),
            quote_locked=float(getattr(q, "locked", 0.0)),
        )
        ctx.repo.append_event("equity.persisted", {"reason": reason, "price": float(px)})
    except Exception as e:
        ctx.repo.append_event("equity.persist_failed", {"reason": reason, "error": str(e)})


def _persist_execution_report(ctx: TradingContext, evt: Dict[str, Any]) -> None:
    """
    Main-thread persistence (safe): events.jsonl + orders.csv/fills.csv if repo supports it.
    """
    ctx.repo.append_event("execution_report", evt)

    try:
        sym = str(evt.get("s") or ctx.symbol)
        side = str(evt.get("S") or "")
        otype = str(evt.get("o") or "")
        status = str(evt.get("X") or "")
        exec_type = str(evt.get("x") or "")
        exchange_oid = str(evt.get("i") or "")
        client_oid = str(evt.get("C") or evt.get("c") or "")

        qty = float(evt.get("q") or 0.0)
        price = float(evt.get("p") or 0.0)

        if hasattr(ctx.repo, "append_order"):
            ctx.repo.append_order(
                symbol=sym,
                side=side,
                order_type=otype,
                client_order_id=client_oid,
                exchange_order_id=exchange_oid,
                status=status,
                qty=qty,
                price=price,
            )

        if exec_type.upper() == "TRADE":
            fill_qty = float(evt.get("l") or 0.0)
            fill_price = float(evt.get("L") or 0.0)
            cum_qty = float(evt.get("z") or 0.0)
            cum_quote = float(evt.get("Z") or 0.0)
            if fill_qty > 0 and fill_price > 0 and hasattr(ctx.repo, "append_fill"):
                ctx.repo.append_fill(
                    symbol=sym,
                    side=side,
                    order_type=otype,
                    client_order_id=client_oid,
                    exchange_order_id=exchange_oid,
                    fill_qty=fill_qty,
                    fill_price=fill_price,
                    cum_qty=cum_qty,
                    cum_quote=cum_quote,
                )
    except Exception:
        # never let persistence break runtime
        pass


def _maybe_enqueue_fill(ctx: TradingContext, evt: Dict[str, Any]) -> None:
    """
    Convert TRADE execution report into OrderFilledEvent and enqueue for strategy processing.
    """
    exec_type_u = str(evt.get("x", "")).upper()
    status_u = str(evt.get("X", "")).upper()

    if exec_type_u != "TRADE":
        return
    if status_u not in {"PARTIALLY_FILLED", "FILLED"}:
        return

    last_qty = to_float(evt.get("l"))
    last_px = to_float(evt.get("L"))
    if last_qty <= 0 or last_px <= 0:
        return

    filled_at = ms_to_dt_utc(evt.get("T", evt.get("E", int(time.time() * 1000))))
    raw_client_oid = str(evt.get("C") or evt.get("c") or "")
    decoded_tag = decode_grid_tag_from_client_order_id(raw_client_oid)
    tag = decoded_tag or raw_client_oid

    side_enum = Side.BUY if str(evt.get("S", "")).upper() == "BUY" else Side.SELL

    ofe = OrderFilledEvent(
        order_id=str(evt.get("i")),
        symbol=str(evt.get("s") or ctx.symbol),
        side=side_enum,
        price=float(last_px),
        qty=float(last_qty),
        filled_at=filled_at,
        client_tag=tag,
    )
    ctx.fill_events.put(ofe)


def pump_async_events(ctx: TradingContext, *, max_reports: int = 1000, max_fills: int = 1000) -> None:
    """
    MAIN THREAD ONLY:
      - drain exec reports from queue
      - update OrderManager
      - persist
      - enqueue fills
      - drain fills and call strategy.on_order_filled sequentially
    """
    # 1) Drain execution reports
    n = 0
    while n < max_reports:
        try:
            evt = ctx.exec_reports.get_nowait()
        except Empty:
            break

        _persist_execution_report(ctx, evt)

        if getattr(ctx, "pnl", None) is not None:
            ctx.pnl.on_execution_report(evt)


        if ctx.order_mgr is not None:
            try:
                ctx.order_mgr.on_execution_report(evt)
            except Exception:
                pass

        _maybe_enqueue_fill(ctx, evt)
        n += 1

    # 2) Drain fills -> strategy
    k = 0
    while k < max_fills:
        try:
            ofe = ctx.fill_events.get_nowait()
        except Empty:
            break

        try:
            ctx.strategy.on_order_filled(ofe)
            ctx.repo.append_event(
                "strategy.on_order_filled",
                {"order_id": ofe.order_id, "tag": ofe.client_tag, "qty": ofe.qty, "price": ofe.price},
            )

            if ctx.pnl_ledger is not None:
                ctx.pnl_ledger.on_fill(ofe)  # fee derived from fee_pct for now
                persist_pnl(ctx, reason="fill", last_price=float(getattr(ctx.equity, "last_price", ofe.price)))

        except Exception:
            pass

        if ctx.equity is not None and hasattr(ctx.equity, "update_on_fill"):
            try:
                ctx.equity.update_on_fill(ofe)
            except Exception:
                pass

        k += 1


def _setup_logging(settings: TradingSettings):
    log_level = str(getattr(settings.run_cfg.logging, "level", "INFO"))
    log_file = init_logging(run_name=settings.run_cfg.name, level_name=log_level)

    log = get_logger("trade")
    ex_log = get_logger("trade.exchange")
    md_log = get_logger("trade.marketdata")
    us_log = get_logger("trade.userstream")
    ord_log = get_logger("trade.orders")

    log.info("=== Trading runtime ===")
    log.info("Run: %s", settings.run_cfg.name)
    log.info("Mode: %s", settings.run_cfg.mode.value)
    log.info("Config: %s", settings.config_path)
    log.info("Log level: %s", log_level)
    log.info("Log file: %s", str(log_file))
    log.info("Execute: %s | Require private: %s", settings.enabled, settings.require_private_endpoints)

    return log, ex_log, md_log, us_log, ord_log, log_level


def _setup_repo(settings: TradingSettings) -> LiveRepository:
    repo = LiveRepository(base_dir="runs/trading", run_name=settings.run_cfg.name)
    run_id = repo.paths.root.name
    repo.write_session(
        {
            "run_id": run_id,
            "run_name": settings.run_cfg.name,
            "mode": settings.run_cfg.mode.value,
            "symbol": settings.run_cfg.data.symbol,
            "interval": settings.run_cfg.data.timeframe,
            "started_at": utcnow_iso(),
            "config_path": settings.config_path,
            "config": settings.raw_cfg,
        }
    )
    return repo


def _setup_exchange(settings: TradingSettings, log, ex_log):
    secrets = EnvSecretsProvider()
    api_key = ""
    api_secret = ""

    try:
        api_key, api_secret = secrets.get_binance_keys(settings.run_cfg.mode)
        log.info("Resolved API key: %s", mask_secret(api_key, keep=6))
        log.info("Resolved API secret: %s", mask_secret(api_secret, keep=2))
    except Exception as e:
        if settings.require_private_endpoints:
            raise RuntimeError(
                f"Missing API credentials via EnvSecretsProvider: {e}. "
                f"For PAPER set BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET. "
                f"For LIVE set BINANCE_API_KEY / BINANCE_API_SECRET."
            ) from e
        log.warning("No private creds (%s). Continuing PUBLIC-ONLY mode.", e)

    exchange = BinanceSpotExchange(
        mode=settings.run_cfg.mode,
        api_key=api_key,
        api_secret=api_secret,
        logger_name="trade.exchange",
    )
    exchange.ping()
    ex_log.info("Binance server time: %s", exchange.server_time())

    return exchange, api_key, api_secret


def _setup_private(ctx: TradingContext) -> None:
    if not ctx.has_private_creds:
        ctx.repo.append_event("private.disabled", {"reason": "missing_creds"})
        return

    balances = ctx.exchange.get_balances(non_zero_only=True)
    ctx.log.info("Account balances (non-zero):")
    assets = list(getattr(ctx.trading, "balances_log_assets", []) or [])
    if assets:
        for a in assets:
            if a in balances:
                free, locked = balances[a]
                ctx.log.info("  %s free=%.8f locked=%.8f", a, free, locked)
    else:
        for asset in sorted(balances.keys()):
            free, locked = balances[asset]
            ctx.log.info("  %s free=%.8f locked=%.8f", asset, free, locked)

    ctx.filters = ctx.exchange.get_symbol_filters(ctx.symbol)
    ctx.log.info(
        "Symbol filters %s: tick_size=%s step_size=%s min_qty=%s min_notional=%s",
        ctx.symbol, ctx.filters.tick_size, ctx.filters.step_size, ctx.filters.min_qty, ctx.filters.min_notional
    )

    open_orders = ctx.exchange.get_open_orders(ctx.symbol)
    ctx.log.info("Open orders: %d", len(open_orders))

    startup = ctx.trading.startup_orders
    cancel_on_startup = bool(getattr(startup, "cancel_open_orders_on_startup", False))
    only_managed = bool(getattr(startup, "cancel_only_managed", True))
    prefixes = tuple(getattr(startup, "cancel_prefixes", []) or ["GT-", "LT-", "LT_"])

    if ctx.settings.run_cfg.mode == RunMode.LIVE and cancel_on_startup and not only_managed:
        raise RuntimeError("Refusing to cancel ALL open orders on LIVE. Keep cancel_only_managed=true.")

    ctx.order_mgr = OrderManager(
        run_name=ctx.settings.run_cfg.name,
        symbol=ctx.symbol,
        exchange=ctx.exchange,
        filters=ctx.filters,
        logger=ctx.ord_log,
    )
    ctx.order_mgr.reconcile_open_orders(open_orders)
    ctx.repo.append_event("orders.reconciled", {"open_orders": int(len(open_orders))})

    if cancel_on_startup and open_orders:
        ctx.log.warning(
            "Startup cleanup: canceling open orders (only_managed=%s prefixes=%s)",
            only_managed, prefixes,
        )
        if only_managed:
            attempted = ctx.order_mgr.cancel_all_managed(reason="startup_cleanup", prefixes=prefixes)
        else:
            attempted = ctx.order_mgr.cancel_all_open_orders(reason="startup_cleanup_all")

        time.sleep(0.2)
        open_orders2 = ctx.exchange.get_open_orders(ctx.symbol)
        ctx.order_mgr.reconcile_open_orders(open_orders2)

        ctx.repo.append_event(
            "startup.cleanup",
            {
                "attempted": int(attempted),
                "only_managed": bool(only_managed),
                "prefixes": list(prefixes),
                "open_orders_after": int(len(open_orders2)),
            },
        )

    try:
        ctx.equity = EquityTracker(exchange=ctx.exchange, symbol=ctx.symbol)

        px, b, q = ctx.equity.snapshot()
        fee_pct = float(getattr(ctx.settings.run_cfg.engine, "trading_fee_pct", 0.0) or 0.0)

        initial_base = float(b.free + b.locked)
        initial_quote = float(q.free + q.locked)

        ctx.pnl_ledger = FillLedger(
            symbol=ctx.symbol,
            fee_pct=fee_pct,
            initial_quote=initial_quote,
            initial_base=initial_base,
            start_price=float(px),
        )

        ctx.buyhold = BuyHoldBaseline(
            initial_quote=initial_quote,
            initial_base=initial_base,
            start_price=float(px),
            fee_pct=fee_pct,
        )

        ctx.repo.append_event(
            "pnl.ledger.init",
            {
                "fee_pct": fee_pct,
                "initial_quote": initial_quote,
                "initial_base": initial_base,
                "start_price": float(px),
            },
        )

        persist_pnl(ctx, reason="startup", last_price=float(px))
        ctx.repo.append_event(
            "equity.snapshot",
            ctx.equity.snapshot_json() if hasattr(ctx.equity, "snapshot_json") else {"price": px},
        )
        ctx.repo.append_equity(
            symbol=ctx.symbol,
            price=float(px),
            base_free=float(getattr(b, "free", 0.0)),
            base_locked=float(getattr(b, "locked", 0.0)),
            quote_free=float(getattr(q, "free", 0.0)),
            quote_locked=float(getattr(q, "locked", 0.0)),
        )
        
        ctx.pnl.seed_from_balance(base_qty=b.total, price=px)
        ctx.repo.append_event("pnl.ledger.init", ctx.pnl.snapshot(price=px).__dict__)

    except Exception as e:
        ctx.log.warning("EquityTracker init failed (%s). Will use REST fallback.", e)
        ctx.equity = None

    ctx.user_stream = start_user_stream(ctx)


def _start_market_stream(ctx: TradingContext) -> BinanceKlineStream:
    use_testnet_ws = resolve_use_testnet_ws(ctx.settings.run_cfg)
    stream = BinanceKlineStream(
        symbol=ctx.symbol,
        interval=ctx.interval,
        use_testnet=use_testnet_ws,
        logger=ctx.md_log,
    )
    stream.start()
    ctx.log.info(
        "Market stream started (symbol=%s interval=%s use_testnet_ws=%s). Waiting for CLOSED candles...",
        ctx.symbol, ctx.interval, use_testnet_ws
    )
    return stream


def _shutdown(ctx: TradingContext) -> None:
    """
    Safe shutdown:
      - never cancels ALL open orders
      - cancels only managed orders (by prefixes) for kill-switch
      - stops streams
    """
    try:
        if ctx.kline_stream is not None:
            ctx.kline_stream.stop()
    except Exception:
        pass

    cancel_on_shutdown = bool(getattr(ctx.trading.safety, "cancel_all_on_shutdown", True))
    if cancel_on_shutdown and ctx.order_mgr is not None and ctx.has_private_creds:
        startup = getattr(ctx.trading, "startup_orders", None)

        prefixes_list = None
        if startup is not None:
            prefixes_list = (
                getattr(startup, "cancel_prefixes", None)
                or getattr(startup, "cancel_open_orders_prefixes", None)
            )

        prefixes = tuple(prefixes_list or ["GT-", "LT-", "LT_"])

        try:
            ctx.log.warning(
                "Kill-switch: canceling MANAGED open orders on shutdown (prefixes=%s)...",
                prefixes,
            )

            attempted = None
            try:
                attempted = ctx.order_mgr.cancel_all_managed(reason="shutdown_killswitch", prefixes=prefixes)
            except TypeError:
                try:
                    attempted = ctx.order_mgr.cancel_all_managed(
                        symbol=ctx.symbol, reason="shutdown_killswitch", prefixes=prefixes
                    )
                except TypeError:
                    attempted = ctx.order_mgr.cancel_all_managed(reason="shutdown_killswitch")

            ctx.repo.append_event(
                "killswitch.cancel_managed",
                {"ok": True, "attempted": int(attempted) if attempted is not None else None, "prefixes": list(prefixes)},
            )

            try:
                time.sleep(0.2)
            except Exception:
                pass

        except Exception as e:
            ctx.repo.append_event(
                "killswitch.cancel_managed",
                {"ok": False, "error": str(e), "prefixes": list(prefixes)},
            )

    try:
        if ctx.user_stream is not None:
            ctx.user_stream.stop()
    except Exception:
        pass


def run_trading(settings: TradingSettings) -> None:
    run_cfg = settings.run_cfg

    log, ex_log, md_log, us_log, ord_log, _ = _setup_logging(settings)
    repo = _setup_repo(settings)

    symbol = run_cfg.data.symbol
    interval = run_cfg.data.timeframe
    base_asset, quote_asset = split_symbol(symbol)

    exchange, api_key, api_secret = _setup_exchange(settings, log, ex_log)
    has_private = bool(api_key) and bool(api_secret)

    from infra.config.strategy_grid import DynamicGridStrategyConfig
    from app.trading.strategy_builder import build_dynamic_grid_strategy

    if not isinstance(run_cfg.strategy, DynamicGridStrategyConfig):
        raise ValueError(
            f"Trading runtime supports only DynamicGridStrategyConfig for now, got: {type(run_cfg.strategy).__name__}"
        )

    strategy: BaseStrategy = build_dynamic_grid_strategy(run_cfg.strategy)
    repo.append_event("strategy.loaded", {"kind": getattr(run_cfg.strategy, "kind", "grid.dynamic"), "symbol": symbol})

    ctx = TradingContext(
        settings=settings,
        symbol=symbol,
        interval=interval,
        base_asset=base_asset,
        quote_asset=quote_asset,
        log=log,
        ex_log=ex_log,
        md_log=md_log,
        us_log=us_log,
        ord_log=ord_log,
        repo=repo,
        exchange=exchange,
        strategy=strategy,
        api_key=api_key,
        api_secret=api_secret,
        has_private_creds=has_private,
    )

    _setup_private(ctx)
    ctx.kline_stream = _start_market_stream(ctx)

    reconcile_interval = float(getattr(ctx.trading, "reconcile_interval_sec", 120) or 0.0)
    last_reconcile_ts = 0.0

    try:
        while True:

            pump_async_events(ctx)

            # Milestone A2: periodic reconciliation (recover from missed stream messages)
            if (
                reconcile_interval > 0
                and ctx.has_private_creds
                and ctx.order_mgr is not None
                and (time.time() - last_reconcile_ts) >= reconcile_interval
            ):
                try:
                    open_orders = ctx.exchange.get_open_orders(ctx.symbol)
                    ctx.order_mgr.reconcile_open_orders(open_orders)
                    ctx.repo.append_event(
                        "orders.reconciled.periodic",
                        {"open_orders": int(len(open_orders)), "interval_sec": reconcile_interval},
                    )
                except Exception as e:
                    ctx.repo.append_event("orders.reconcile_failed", {"where": "periodic", "error": str(e)})

                last_reconcile_ts = time.time()

            try:
                candle = ctx.kline_stream.next_closed_candle(timeout_sec=2.0)
            except TimeoutError:
                continue

            pump_async_events(ctx)

            ctx.log.info(
                "CLOSED candle %s O=%.8f H=%.8f H=%.8f L=%.8f C=%.8f V=%.8f",
                candle.timestamp.isoformat(),
                candle.open, candle.high, candle.high, candle.low, candle.close, candle.volume
            )
            ctx.repo.append_event(
                "candle.closed",
                {
                    "ts": candle.timestamp.isoformat(),
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                },
            )
            
            if ctx.equity is not None:
                ctx.equity.refresh(last_price=float(candle.close))

            persist_equity(ctx, reason="candle_closed", last_price=float(candle.close))
            persist_pnl(ctx, reason="candle_closed", last_price=float(candle.close))

            if not ctx.has_private_creds:
                ctx.repo.append_event("strategy.skipped", {"reason": "missing_private_creds"})
                continue

            account_state = build_account_state(
                candle=candle,
                base_asset=ctx.base_asset,
                quote_asset=ctx.quote_asset,
                exchange=ctx.exchange,
                equity_tracker=ctx.equity,
            )
            if account_state is None:
                ctx.repo.append_event("strategy.skipped", {"reason": "no_account_state"})
                continue

            actions = ctx.strategy.on_candle(candle, account_state) or []
            ctx.repo.append_event("strategy.actions", {"n": len(actions), "actions": [str(a) for a in actions]})

            if not settings.enabled:
                continue

            if ctx.order_mgr is None:
                ctx.repo.append_event("trade.blocked", {"reason": "order_mgr_missing"})
                continue

            execute_actions(
                actions=actions,
                candle=candle,
                exchange=ctx.exchange,
                order_mgr=ctx.order_mgr,
                safety=ctx.trading.safety,
                base_asset=ctx.base_asset,
                log=ctx.log,
                repo=ctx.repo,
            )

            persist_equity(ctx, reason="after_execute_actions", last_price=float(candle.close))

    except KeyboardInterrupt:
        ctx.log.info("Stopping trading runtime (KeyboardInterrupt)...")
        ctx.repo.append_event("runtime.stop", {"reason": "KeyboardInterrupt"})
    finally:
        _shutdown(ctx)
