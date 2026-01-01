# core/live/pnl_ledger.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.models import OrderFilledEvent, Side


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Lot:
    qty: float
    entry_price: float
    entry_fee_quote: float
    opened_at: datetime
    tag: str


class BuyHoldBaseline:
    """
    Buy&Hold baseline:
      - at start_price: buy base with ALL quote once (fee applied)
      - hold until last_price
    """
    def __init__(self, *, initial_quote: float, initial_base: float, start_price: float, fee_pct: float) -> None:
        self.initial_quote = float(initial_quote)
        self.initial_base = float(initial_base)
        self.start_price = float(start_price)
        self.fee_pct = max(0.0, float(fee_pct))

        if self.start_price > 0.0 and self.initial_quote > 0.0:
            bought_qty = self.initial_quote / (self.start_price * (1.0 + self.fee_pct))
        else:
            bought_qty = 0.0

        self.base_total = float(self.initial_base + bought_qty)
        self.initial_equity = float(self.base_total * self.start_price)

    def equity(self, *, last_price: float) -> float:
        return float(self.base_total * float(last_price))

    def return_pct(self, *, last_price: float) -> float:
        eq = self.equity(last_price=float(last_price))
        if self.initial_equity <= 0.0:
            return 0.0
        return float((eq / self.initial_equity - 1.0) * 100.0)


class FillLedger:
    """
    Lightweight live/paper ledger:
    - tracks cash (quote) + open lots (base) by *tag*
    - computes realized pnl on sells (prefers same-tag lots)
    - computes total pnl by mark-to-market using last_price
    """
    def __init__(
        self,
        *,
        symbol: str,
        fee_pct: float,
        initial_quote: float,
        initial_base: float,
        start_price: float,
        allow_cross_tag_close: bool = True,
    ) -> None:
        self.symbol = symbol
        self.fee_pct = max(0.0, float(fee_pct))

        self.initial_quote = float(initial_quote)
        self.initial_base = float(initial_base)
        self.start_price = float(start_price)

        self.cash_quote = float(initial_quote)
        self.fees_paid_quote = 0.0
        self.realized_pnl_quote = 0.0

        self._lots_by_tag: Dict[str, List[Lot]] = {}
        self._allow_cross_tag_close = bool(allow_cross_tag_close)

        # Treat starting base inventory as an initial lot so sells are well-defined.
        if self.initial_base > 0.0 and self.start_price > 0.0:
            self._lots_by_tag["__INITIAL__"] = [
                Lot(
                    qty=float(self.initial_base),
                    entry_price=float(self.start_price),
                    entry_fee_quote=0.0,
                    opened_at=_utcnow(),
                    tag="__INITIAL__",
                )
            ]

    def _fee_quote(self, *, price: float, qty: float) -> float:
        return float(float(price) * float(qty) * self.fee_pct)

    def _base_position(self) -> float:
        return float(sum(l.qty for lots in self._lots_by_tag.values() for l in lots))

    def _invested_cost_quote(self) -> float:
        # cost basis for open lots incl. entry fees already paid
        return float(
            sum(l.qty * l.entry_price + l.entry_fee_quote for lots in self._lots_by_tag.values() for l in lots)
        )

    def _pop_oldest_lot(self) -> Optional[Tuple[str, int, Lot]]:
        oldest: Optional[Tuple[str, int, Lot]] = None
        for tag, lots in self._lots_by_tag.items():
            for i, lot in enumerate(lots):
                if oldest is None or lot.opened_at < oldest[2].opened_at:
                    oldest = (tag, i, lot)
        return oldest

    def on_fill(self, event: OrderFilledEvent, *, fee_quote: Optional[float] = None) -> None:
        if float(event.qty) <= 0.0 or float(event.price) <= 0.0:
            return

        tag = str(event.client_tag or "").strip() or "__UNSPEC__"
        qty = float(event.qty)
        px = float(event.price)

        fee = float(fee_quote) if (fee_quote is not None and float(fee_quote) >= 0.0) else self._fee_quote(price=px, qty=qty)
        self.fees_paid_quote += fee

        if event.side == Side.BUY:
            # cash outflow includes fee (assume fee in quote)
            self.cash_quote -= (px * qty + fee)
            self._lots_by_tag.setdefault(tag, []).append(
                Lot(qty=qty, entry_price=px, entry_fee_quote=fee, opened_at=_utcnow(), tag=tag)
            )
            return

        # SELL
        self.cash_quote += (px * qty - fee)

        remaining = qty
        total_sell_qty = qty

        while remaining > 1e-12:
            lots = self._lots_by_tag.get(tag, [])

            pick: Optional[Tuple[str, int, Lot]] = None
            if lots:
                pick = (tag, 0, lots[0])
            elif self._allow_cross_tag_close:
                pick = self._pop_oldest_lot()

            if pick is None:
                # Nothing to close against -> treat as unknown basis (no realized pnl update)
                break

            pick_tag, idx, lot = pick

            close_qty = min(remaining, float(lot.qty))
            if close_qty <= 0.0:
                break

            # allocate fees proportionally
            buy_fee_part = float(lot.entry_fee_quote) * (close_qty / float(lot.qty)) if lot.qty > 0 else 0.0
            sell_fee_part = float(fee) * (close_qty / total_sell_qty) if total_sell_qty > 0 else 0.0

            gross = (px - float(lot.entry_price)) * close_qty
            realized = gross - buy_fee_part - sell_fee_part
            self.realized_pnl_quote += realized

            # shrink lot
            lot.qty = float(lot.qty) - close_qty
            lot.entry_fee_quote = float(lot.entry_fee_quote) - buy_fee_part

            # remove if depleted
            if lot.qty <= 1e-12:
                self._lots_by_tag[pick_tag].pop(idx)
                if not self._lots_by_tag[pick_tag]:
                    self._lots_by_tag.pop(pick_tag, None)

            remaining -= close_qty

    def snapshot(self, *, last_price: float) -> dict:
        lp = float(last_price)
        base_pos = self._base_position()
        ledger_equity = float(self.cash_quote + base_pos * lp)

        initial_equity = float(self.initial_quote + self.initial_base * self.start_price) if self.start_price > 0 else float(self.initial_quote)
        total_pnl = float(ledger_equity - initial_equity)

        # unrealized vs realized split (unrealized computed from open lots mark-to-market)
        unrealized = 0.0
        for lots in self._lots_by_tag.values():
            for l in lots:
                unrealized += (lp - float(l.entry_price)) * float(l.qty)

        return {
            "cash_quote": float(self.cash_quote),
            "base_position": float(base_pos),
            "invested_cost_quote": float(self._invested_cost_quote()),
            "fees_paid_quote": float(self.fees_paid_quote),
            "realized_pnl_quote": float(self.realized_pnl_quote),
            "unrealized_pnl_quote": float(unrealized),
            "ledger_equity_quote": float(ledger_equity),
            "initial_equity_quote": float(initial_equity),
            "total_pnl_quote": float(total_pnl),
        }

    def seed_from_balance(self, *, base_qty: float, price: float) -> None:
        base_qty = float(base_qty)
        price = float(price)
        if price <= 0.0:
            raise ValueError("seed_from_balance: price must be > 0")

        # Reset starting point to current wallet state
        self.initial_base = base_qty
        self.start_price = price
        self._base_position = base_qty

        self._lots_by_tag.clear()
        if base_qty > 0.0:
            self._lots_by_tag["__seed__"] = [
                Lot(qty=base_qty, entry_price=price, entry_fee_quote=0.0, opened_at=self.opened_at)
            ]
