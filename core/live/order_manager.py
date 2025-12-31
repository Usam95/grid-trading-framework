# core/live/order_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Grid tag format used by strategies: "XRPUSDT:LVL:3"
_GRID_TAG_RE = re.compile(r"^(?P<sym>[A-Z0-9]+):LVL:(?P<lvl>\d+)$")


def _short_hash(s: str, n: int = 4) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def encode_grid_client_order_id(
    *,
    run_name: str,
    symbol: str,
    level: int,
    side: str,
    salt: str = "",
) -> str:
    """
    Parseable, self-contained clientOrderId (restart-safe).

    Example:
      LT-XRPUSDT-L03-B-a1b2
      LT-XRPUSDT-L03-S-a1b2

    - level is 0-based index
    - side: BUY/SELL -> B/S
    - suffix makes it unique/idempotent per (run+order params)
    """
    sym = symbol.upper()
    side_c = "B" if side.upper() == "BUY" else "S"
    suffix = _short_hash(f"{run_name}|{sym}|{level}|{side_c}|{salt}", 4)
    return f"LT-{sym}-L{level:02d}-{side_c}-{suffix}"


def decode_grid_tag_from_client_order_id(client_order_id: str) -> Optional[str]:
    """
    Returns a grid tag like: XRPUSDT:LVL:3
    or None if not a grid-encoded id.
    """
    if not client_order_id:
        return None

    s = client_order_id.strip().upper()

    # Already a grid tag?
    m = _GRID_TAG_RE.match(s)
    if m:
        return f"{m.group('sym')}:LVL:{int(m.group('lvl'))}"

    # Encoded format: LT-SYMBOL-L03-B-a1b2
    parts = s.split("-")
    if len(parts) >= 5 and parts[0] == "LT":
        sym = parts[1]
        lvl_part = parts[2]  # e.g. L03
        if lvl_part.startswith("L") and lvl_part[1:].isdigit():
            lvl = int(lvl_part[1:])
            return f"{sym}:LVL:{lvl}"

    return None


@dataclass
class ManagedOrder:
    symbol: str
    side: str
    order_type: str
    qty: float
    price: Optional[float]

    client_order_id: str
    exchange_order_id: Optional[int] = None

    status: str = "NEW"
    cum_qty: float = 0.0
    cum_quote: float = 0.0
    last_fill_qty: float = 0.0
    last_fill_price: float = 0.0

    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    # Whether we believe this order was created by *this* runtime (vs reconciled/foreign).
    managed: bool = True

    def is_terminal(self) -> bool:
        return self.status.upper() in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}


class OrderManager:
    """
    B6: The "brainstem" between strategy/harness and the exchange.
      - idempotent clientOrderId generation
      - tracks open orders locally
      - reconciliation on startup from get_open_orders()
      - consumes executionReport events to update state
    """

    def __init__(self, *, run_name: str, symbol: str, exchange: Any, filters: Any, logger: Any) -> None:
        self.run_name = run_name
        self.symbol = symbol.upper()
        self.exchange = exchange
        self.filters = filters
        self.log = logger

        self._by_client_id: Dict[str, ManagedOrder] = {}
        self._by_order_id: Dict[int, ManagedOrder] = {}

    # -----------------------
    # ClientOrderId (idempotency)
    # -----------------------
    def make_client_order_id(
        self,
        *,
        intent: str,
        side: str,
        order_type: str,
        qty: float,
        price: Optional[float],
        tag: str = "",
    ) -> str:
        """
        Option A: If intent == "grid" AND tag matches "<SYMBOL>:LVL:<n>",
        encode the grid identity into clientOrderId so it is parseable and restart-safe.

        Otherwise fallback to the legacy GT-<hash> ids.
        """
        intent_l = (intent or "").strip().lower()
        tag_u = (tag or "").strip().upper()

        # Grid strategy orders: produce parseable LT-... ids
        if intent_l == "grid":
            m = _GRID_TAG_RE.match(tag_u)
            if m:
                lvl = int(m.group("lvl"))
                # include order params in salt to make the id idempotent per (type/px/qty)
                p = "" if price is None else f"{float(price):.8f}"
                salt = f"{order_type.upper()}|{p}|{float(qty):.8f}"
                return encode_grid_client_order_id(
                    run_name=self.run_name,
                    symbol=self.symbol,
                    level=lvl,
                    side=side,
                    salt=salt,
                )

        # Legacy deterministic hash (good for B3/B5 harness)
        p = "" if price is None else f"{float(price):.8f}"
        canon = (
            f"{self.run_name}|{self.symbol}|{intent}|{side.upper()}|{order_type.upper()}|"
            f"{p}|{float(qty):.8f}|{tag}"
        )
        h = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:26]
        return f"GT-{h}"

    def count_open(self, *, managed_only: bool = True, prefixes: tuple[str, ...] = ("LT-", "GT-", "LT_")) -> int:
        n = 0
        for mo in self._by_client_id.values():
            if mo.is_terminal():
                continue
            if managed_only and not mo.client_order_id.startswith(prefixes):
                continue
            n += 1
        return n

    def cancel_all_managed(
        self,
        *,
        reason: str = "startup_cleanup",
        prefixes: tuple[str, ...] = ("LT-", "GT-", "LT_"),
    ) -> int:
        """
        Cancel all NON-terminal orders tracked by this OrderManager whose clientOrderId starts with prefixes.
        Returns number of cancel attempts (not necessarily successes).
        """
        to_cancel: list[ManagedOrder] = []
        for mo in self._by_client_id.values():
            if mo.is_terminal():
                continue
            if not mo.client_order_id.startswith(prefixes):
                continue
            to_cancel.append(mo)

        self.log.warning("cancel_all_managed reason=%s count=%d prefixes=%s", reason, len(to_cancel), prefixes)

        attempted = 0
        for mo in to_cancel:
            attempted += 1
            try:
                # Prefer cancel by exchange order id when we have it, otherwise by client id
                self.cancel(order_id=mo.exchange_order_id, client_order_id=None if mo.exchange_order_id else mo.client_order_id)
                self.log.info("canceled clientId=%s oid=%s reason=%s", mo.client_order_id, mo.exchange_order_id, reason)
            except Exception as e:
                self.log.warning("failed to cancel clientId=%s oid=%s: %s", mo.client_order_id, mo.exchange_order_id, e)
        return attempted

    def cancel_all_open_orders(self, *, reason: str = "startup_cleanup_all") -> int:
        """
        Dangerous helper: cancel ALL tracked NON-terminal orders (managed or not).
        """
        to_cancel: list[ManagedOrder] = []
        for mo in self._by_client_id.values():
            if not mo.is_terminal():
                to_cancel.append(mo)

        self.log.warning("cancel_all_open_orders reason=%s count=%d", reason, len(to_cancel))

        attempted = 0
        for mo in to_cancel:
            attempted += 1
            try:
                self.cancel(order_id=mo.exchange_order_id, client_order_id=None if mo.exchange_order_id else mo.client_order_id)
            except Exception as e:
                self.log.warning("failed to cancel clientId=%s oid=%s: %s", mo.client_order_id, mo.exchange_order_id, e)
        return attempted


    # -----------------------
    # Reconciliation
    # -----------------------
    def reconcile_open_orders(self, open_orders: list[dict[str, Any]]) -> None:
        """
        Call once on startup: rebuild local open-order state from the exchange.
        """
        self._by_client_id.clear()
        self._by_order_id.clear()

        for o in open_orders:
            try:
                client_id = str(o.get("clientOrderId") or o.get("origClientOrderId") or "")
                order_id = o.get("orderId")
                side = str(o.get("side") or "").upper()
                typ = str(o.get("type") or "").upper()
                status = str(o.get("status") or "NEW").upper()
                price = float(o.get("price") or 0.0)
                qty = float(o.get("origQty") or 0.0)
                exec_qty = float(o.get("executedQty") or 0.0)
                cum_quote = float(o.get("cummulativeQuoteQty") or 0.0)
            except Exception:
                continue

            mo = ManagedOrder(
                symbol=self.symbol,
                side=side,
                order_type=typ,
                qty=qty,
                price=price if price > 0 else None,
                client_order_id=client_id if client_id else f"UNKNOWN-{order_id}",
                exchange_order_id=int(order_id) if isinstance(order_id, (int, str)) and str(order_id).isdigit() else None,
                status=status,
                cum_qty=exec_qty,
                cum_quote=cum_quote,
                managed=bool(client_id.startswith(("LT-", "GT-"))) if client_id else False,
            )

            self._by_client_id[mo.client_order_id] = mo
            if mo.exchange_order_id is not None:
                self._by_order_id[mo.exchange_order_id] = mo

        self.log.debug("reconciled open orders: %d", len(self._by_client_id))

    # -----------------------
    # Submission helpers
    # -----------------------
    def place_limit(
        self,
        *,
        side: str,
        qty: float,
        price: float,
        intent: str,
        tag: str = "",
    ) -> ManagedOrder:
        """
        Place LIMIT in an idempotent way.
        If we already track an active order for the same client id -> return it.
        """
        client_id = self.make_client_order_id(
            intent=intent,
            side=side,
            order_type="LIMIT",
            qty=qty,
            price=price,
            tag=tag,
        )

        existing = self._by_client_id.get(client_id)
        if existing and not existing.is_terminal():
            self.log.info(
                "idempotent hit -> reuse existing order clientId=%s status=%s",
                client_id,
                existing.status,
            )
            return existing

        # Pre-register as PENDING so executionReport NEW won't be "unknown" due to races
        pending = ManagedOrder(
            symbol=self.symbol,
            side=side.upper(),
            order_type="LIMIT",
            qty=qty,
            price=price,
            client_order_id=client_id,
            exchange_order_id=None,
            status="PENDING_SUBMIT",
            managed=True,
        )
        self._by_client_id[client_id] = pending

        try:
            ack = self.exchange.place_limit_order(
                symbol=self.symbol,
                side=side,
                quantity=qty,
                price=price,
                client_order_id=client_id,
                filters=self.filters,
            )
        except Exception as e:
            self.log.warning(
                "place_limit failed (clientId=%s): %s -> trying get_order by client id",
                client_id,
                e,
            )
            ack = self.exchange.get_order(self.symbol, client_order_id=client_id)

        order_id = ack.get("orderId")
        status = str(ack.get("status") or "NEW").upper()
        exec_qty = float(ack.get("executedQty") or 0.0)
        cum_quote = float(ack.get("cummulativeQuoteQty") or 0.0)

        mo = ManagedOrder(
            symbol=self.symbol,
            side=side.upper(),
            order_type="LIMIT",
            qty=float(ack.get("origQty") or qty),
            price=float(ack.get("price") or price),
            client_order_id=client_id,
            exchange_order_id=int(order_id) if isinstance(order_id, (int, str)) and str(order_id).isdigit() else None,
            status=status,
            cum_qty=exec_qty,
            cum_quote=cum_quote,
            managed=True,
        )

        self._by_client_id[client_id] = mo
        if mo.exchange_order_id is not None:
            self._by_order_id[mo.exchange_order_id] = mo

        return mo

    def place_market(
        self,
        *,
        side: str,
        qty: float,
        intent: str,
        tag: str = "",
    ) -> ManagedOrder:
        """
        Place MARKET in an idempotent way + avoid WS race by pre-registering PENDING_SUBMIT.
        """
        client_id = self.make_client_order_id(
            intent=intent,
            side=side,
            order_type="MARKET",
            qty=qty,
            price=None,
            tag=tag,
        )

        existing = self._by_client_id.get(client_id)
        if existing and not existing.is_terminal():
            self.log.info(
                "idempotent hit -> reuse existing MARKET clientId=%s status=%s",
                client_id,
                existing.status,
            )
            return existing

        pending = ManagedOrder(
            symbol=self.symbol,
            side=side.upper(),
            order_type="MARKET",
            qty=qty,
            price=None,
            client_order_id=client_id,
            exchange_order_id=None,
            status="PENDING_SUBMIT",
            managed=True,
        )
        self._by_client_id[client_id] = pending

        try:
            ack = self.exchange.place_market_order(
                symbol=self.symbol,
                side=side,
                quantity=qty,
                client_order_id=client_id,  # adapter must support this
                filters=self.filters,
            )
        except Exception as e:
            self.log.warning(
                "place_market failed (clientId=%s): %s -> trying get_order by client id",
                client_id,
                e,
            )
            ack = self.exchange.get_order(self.symbol, client_order_id=client_id)

        order_id = ack.get("orderId")
        status = str(ack.get("status") or "NEW").upper()
        exec_qty = float(ack.get("executedQty") or 0.0)
        cum_quote = float(ack.get("cummulativeQuoteQty") or 0.0)

        mo = ManagedOrder(
            symbol=self.symbol,
            side=side.upper(),
            order_type="MARKET",
            qty=float(ack.get("origQty") or qty),
            price=None,
            client_order_id=client_id,
            exchange_order_id=int(order_id) if isinstance(order_id, (int, str)) and str(order_id).isdigit() else None,
            status=status,
            cum_qty=exec_qty,
            cum_quote=cum_quote,
            managed=True,
        )

        self._by_client_id[client_id] = mo
        if mo.exchange_order_id is not None:
            self._by_order_id[mo.exchange_order_id] = mo
        return mo

    def cancel(self, *, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> dict[str, Any]:
        return self.exchange.cancel_order(
            symbol=self.symbol,
            order_id=order_id,
            client_order_id=client_order_id,
        )

    # -----------------------
    # ExecutionReport consumer
    # -----------------------
    def on_execution_report(self, evt: dict[str, Any]) -> None:
        """
        Binance executionReport fields we care about:
          s=symbol, i=orderId, c=clientOrderId, C=origClientOrderId, X=status,
          l=lastFilledQty, L=lastFilledPrice, z=cumQty, Z=cumQuote
        """
        sym = str(evt.get("s") or "").upper()
        if sym and sym != self.symbol:
            return

        cid = str(evt.get("c") or "")
        ocid = str(evt.get("C") or "")

        oid_raw = evt.get("i")
        oid = int(oid_raw) if isinstance(oid_raw, (int, str)) and str(oid_raw).isdigit() else None

        status = str(evt.get("X") or "").upper()
        last_fill_qty = float(evt.get("l") or 0.0)
        last_fill_price = float(evt.get("L") or 0.0)
        cum_qty = float(evt.get("z") or 0.0)
        cum_quote = float(evt.get("Z") or 0.0)

        mo: Optional[ManagedOrder] = None

        # 1) Prefer exact client ids first
        if cid:
            mo = self._by_client_id.get(cid)
        if mo is None and ocid:
            mo = self._by_client_id.get(ocid)

        # 2) Fallback by exchange order id
        if mo is None and oid is not None:
            mo = self._by_order_id.get(oid)

        # 3) Alias both cid/ocid to same object
        if mo is not None:
            if cid:
                self._by_client_id[cid] = mo
            if ocid:
                self._by_client_id[ocid] = mo

        if mo is not None and mo.exchange_order_id is None and oid is not None:
            mo.exchange_order_id = oid
            self._by_order_id[oid] = mo

        if mo is None:
            best_client_id = ocid or cid or f"UNKNOWN-{oid}"
            mo = ManagedOrder(
                symbol=self.symbol,
                side=str(evt.get("S") or "").upper(),
                order_type=str(evt.get("o") or "").upper(),
                qty=float(evt.get("q") or 0.0),
                price=float(evt.get("p") or 0.0) if evt.get("p") else None,
                client_order_id=best_client_id,
                exchange_order_id=oid,
                status=status or "NEW",
                managed=bool(best_client_id.startswith(("LT-", "GT-"))),
            )
            self._by_client_id[mo.client_order_id] = mo
            if oid is not None:
                self._by_order_id[oid] = mo

            self.log.warning(
                "executionReport for unknown order -> now tracking clientId=%s oid=%s",
                mo.client_order_id,
                oid,
            )

        mo.status = status or mo.status
        mo.last_fill_qty = last_fill_qty
        mo.last_fill_price = last_fill_price
        mo.cum_qty = cum_qty
        mo.cum_quote = cum_quote
        mo.updated_at = _utcnow()

        if mo.is_terminal():
            self.log.info(
                "order terminal clientId=%s status=%s cumQty=%.8f",
                mo.client_order_id,
                mo.status,
                mo.cum_qty,
            )
