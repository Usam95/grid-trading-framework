# core/execution/reservations.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.models import Order, Side


@dataclass(frozen=True)
class Reservation:
    quote: float = 0.0  # locked quote currency
    base: float = 0.0   # locked base quantity


@dataclass
class ReservationBook:
    """
    Tracks reserved funds for open orders (spot-style locking).

    - For BUY orders: reserve quote = price * qty * (1 + fee_pct)
    - For SELL orders: reserve base = qty
    """
    enabled: bool
    fee_pct: float

    _by_order_id: Dict[str, Reservation] = field(default_factory=dict)
    reserved_quote: float = 0.0
    reserved_base: float = 0.0

    def compute(self, order: Order) -> Reservation:
        if not self.enabled:
            return Reservation()

        if order.side == Side.BUY:
            price = float(order.price or 0.0)
            qty = float(order.qty or 0.0)
            if price <= 0 or qty <= 0:
                return Reservation()
            quote = price * qty * (1.0 + self.fee_pct)
            return Reservation(quote=quote, base=0.0)

        qty = float(order.qty or 0.0)
        if qty <= 0:
            return Reservation()
        return Reservation(quote=0.0, base=qty)

    def reserve(self, order: Order) -> Reservation:
        if not self.enabled:
            return Reservation()

        if not order.id:
            raise ValueError("Cannot reserve funds: order.id is empty.")

        res = self.compute(order)

        if res.quote or res.base:
            if order.id in self._by_order_id:
                self.release(order.id)

            self._by_order_id[order.id] = res
            self.reserved_quote += res.quote
            self.reserved_base += res.base

        return res

    def release(self, order_id: str) -> Reservation:
        if not self.enabled:
            return Reservation()

        res = self._by_order_id.pop(order_id, None)
        if res is None:
            return Reservation()

        self.reserved_quote -= res.quote
        self.reserved_base -= res.base

        self.reserved_quote = max(0.0, self.reserved_quote)
        self.reserved_base = max(0.0, self.reserved_base)

        return res
