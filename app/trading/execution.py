# app/trading/execution.py
from __future__ import annotations

from typing import Any, Iterable, Optional

from core.engine_actions import EngineAction, EngineActionType
from core.models import Candle, OrderType, Side


def count_managed_open_orders(order_mgr) -> int:
    # support older/internal attribute names defensively
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


def cancel_all_managed(order_mgr, *, symbol: str, reason: str) -> None:
    """
    Kill-switch cancel. Prefer OrderManager.cancel_all_managed().
    """
    if hasattr(order_mgr, "cancel_all_managed"):
        try:
            order_mgr.cancel_all_managed(reason=reason)
            return
        except TypeError:
            # some variants require symbol or prefixes
            pass

    for meth in ("cancel_all", "cancel_all_open", "cancel_all_open_orders"):
        if hasattr(order_mgr, meth):
            getattr(order_mgr, meth)(symbol=symbol, reason=reason)  # type: ignore
            return

    raise RuntimeError("OrderManager has no cancel_all_* method; implement one for kill-switch safety.")


def exec_place_order(
    *,
    action: EngineAction,
    candle: Candle,
    exchange,
    order_mgr,
    safety: Optional[Any],
    base_asset: str,
    log,
    repo,
) -> None:
    """
    Execute a single PLACE_ORDER action via OrderManager with safety checks.
    """
    order = action.order
    if order is None:
        return

    px = float(order.price or 0.0)
    if order.type == OrderType.MARKET:
        px = float(candle.close)

    qty = float(order.qty or 0.0)
    if qty <= 0:
        repo.append_event("trade.refused", {"reason": "qty<=0", "client_tag": order.client_tag})
        return

    # --- safety: max notional
    max_notional = float(getattr(safety, "max_notional_per_order", 0.0) or 0.0) if safety is not None else 0.0
    notional = px * qty
    if max_notional > 0 and notional > max_notional:
        repo.append_event(
            "trade.refused",
            {"reason": "max_notional_per_order", "notional": notional, "limit": max_notional, "client_tag": order.client_tag},
        )
        return

    # --- safety: max open orders
    max_open = int(getattr(safety, "max_open_orders", 0) or 0) if safety is not None else 0
    if max_open > 0:
        n_open = count_managed_open_orders(order_mgr)
        if n_open >= max_open:
            repo.append_event(
                "trade.refused",
                {"reason": "max_open_orders", "open_orders": n_open, "limit": max_open, "client_tag": order.client_tag},
            )
            return

    # --- safety: position cap (spot base holdings)
    max_pos_base = float(getattr(safety, "max_position_base", 0.0) or 0.0) if safety is not None else 0.0
    if max_pos_base > 0 and order.side == Side.BUY:
        balances = exchange.get_balances(non_zero_only=False)
        base_free, base_locked = balances.get(base_asset, (0.0, 0.0))
        base_total = float(base_free) + float(base_locked)
        if (base_total + qty) > max_pos_base:
            repo.append_event(
                "trade.refused",
                {"reason": "max_position_base", "base_total": base_total, "buy_qty": qty, "limit": max_pos_base, "client_tag": order.client_tag},
            )
            return

    side_u = order.side.value.upper()

    # Submit via OrderManager
    if order.type == OrderType.MARKET:
        res = order_mgr.place_market(side=side_u, qty=qty, intent="grid", tag=order.client_tag)
        repo.append_event("trade.submitted", {"type": "MARKET", "side": side_u, "qty": qty, "client_tag": order.client_tag, "result": str(res)})
        log.info("SUBMIT: MARKET %s qty=%.8f tag=%s", side_u, qty, order.client_tag)

    elif order.type == OrderType.LIMIT:
        if px <= 0:
            repo.append_event("trade.refused", {"reason": "limit_px<=0", "client_tag": order.client_tag})
            return
        res = order_mgr.place_limit(side=side_u, qty=qty, price=px, intent="grid", tag=order.client_tag)
        repo.append_event("trade.submitted", {"type": "LIMIT", "side": side_u, "qty": qty, "price": px, "client_tag": order.client_tag, "result": str(res)})
        log.info("SUBMIT: LIMIT %s qty=%.8f price=%.8f tag=%s", side_u, qty, px, order.client_tag)

    else:
        repo.append_event("trade.refused", {"reason": f"unsupported_order_type:{order.type}", "client_tag": order.client_tag})


def execute_actions(
    *,
    actions: Iterable[EngineAction],
    candle: Candle,
    exchange,
    order_mgr,
    safety: Optional[Any],
    base_asset: str,
    log,
    repo,
) -> None:
    """
    Execute strategy actions (only PLACE_ORDER currently).
    """
    for action in actions:
        if action.type != EngineActionType.PLACE_ORDER:
            repo.append_event("trade.ignored", {"type": str(action.type), "reason": "execute_only_place_order"})
            continue

        exec_place_order(
            action=action,
            candle=candle,
            exchange=exchange,
            order_mgr=order_mgr,
            safety=safety,
            base_asset=base_asset,
            log=log,
            repo=repo,
        )
