# app/trading/execution.py
from __future__ import annotations

from typing import Any, Iterable, Optional

from core.engine_actions import EngineAction, EngineActionType
from core.models import Candle, OrderType, Side



def _extract_order_ids(res: Any) -> tuple[str, str]:
    """
    Try to extract (exchange_order_id, client_order_id) from OrderManager REST result.
    Works with dicts and simple objects.
    """
    if res is None:
        return "", ""

    exchange_oid = ""
    client_oid = ""

    if isinstance(res, dict):
        exchange_oid = str(res.get("orderId") or res.get("order_id") or res.get("id") or "")
        client_oid = str(res.get("clientOrderId") or res.get("client_order_id") or res.get("clientOrderID") or "")
        return exchange_oid, client_oid

    # object-like
    for k in ("orderId", "order_id", "id"):
        if hasattr(res, k):
            exchange_oid = str(getattr(res, k) or "")
            break
    for k in ("clientOrderId", "client_order_id", "clientOrderID"):
        if hasattr(res, k):
            client_oid = str(getattr(res, k) or "")
            break

    return exchange_oid, client_oid


def _persist_submitted_intent(
    *,
    repo: Any,
    symbol: str,
    side: str,
    order_type: str,
    px: float,
    qty: float,
    client_tag: str,
    res: Any,
) -> None:
    """
    Milestone A1:
    Persist a durable trace immediately after successful REST submit,
    BEFORE any execution reports arrive.
    """
    if not hasattr(repo, "append_order"):
        return

    exchange_oid, client_oid = _extract_order_ids(res)

    # We strongly prefer the true exchange clientOrderId.
    # If it is missing, we still write a row (client id empty) and keep the tag in events.
    repo.append_order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        client_order_id=client_oid,
        exchange_order_id=exchange_oid,
        status="SUBMITTED",
        qty=float(qty),
        price=float(px),
    )

    repo.append_event(
        "order.submitted_intent",
        {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "qty": float(qty),
            "price": float(px),
            "client_tag": client_tag,
            "exchange_order_id": exchange_oid,
            "client_order_id": client_oid,
        },
    )


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

        _persist_submitted_intent(
            repo=repo,
            symbol=str(getattr(exchange, "symbol", "") or action.order.symbol or ""),  # safe fallback
            side=side_u,
            order_type="MARKET",
            px=float(px),
            qty=float(qty),
            client_tag=str(order.client_tag),
            res=res,
        )

        repo.append_event(
            "trade.submitted",
            {"type": "MARKET", "side": side_u, "qty": qty, "client_tag": order.client_tag, "result": str(res)},
        )
        log.info("SUBMIT: MARKET %s qty=%.8f tag=%s", side_u, qty, order.client_tag)

    elif order.type == OrderType.LIMIT:
        if px <= 0:
            repo.append_event("trade.refused", {"reason": "limit_px<=0", "client_tag": order.client_tag})
            return

        res = order_mgr.place_limit(side=side_u, qty=qty, price=px, intent="grid", tag=order.client_tag)

        _persist_submitted_intent(
            repo=repo,
            symbol=str(getattr(exchange, "symbol", "") or action.order.symbol or ""),  # safe fallback
            side=side_u,
            order_type="LIMIT",
            px=float(px),
            qty=float(qty),
            client_tag=str(order.client_tag),
            res=res,
        )

        repo.append_event(
            "trade.submitted",
            {"type": "LIMIT", "side": side_u, "qty": qty, "price": px, "client_tag": order.client_tag, "result": str(res)},
        )
        log.info("SUBMIT: LIMIT %s qty=%.8f price=%.8f tag=%s", side_u, qty, px, order.client_tag)



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
