# core/results/trade_builder.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from core.models import OrderFilledEvent, Side
from core.results.models import Trade


@dataclass
class _OpenLot:
    entry_time: datetime
    entry_price: float
    size: float
    entry_fee: float  # buy-fee remaining (reduced proportionally on partial closes)


class TradeBuilder:
    """
    Builds completed Trade objects from OrderFilledEvents.

    Supports:
      - FIFO lot matching (BUY opens lots, SELL closes lots)
      - partial closes
      - one SELL closing multiple lots
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._open_lots: List[_OpenLot] = []
        self._trades: List[Trade] = []
        self._counter = 0

    def on_fill(self, event: OrderFilledEvent, fee: float) -> None:
        if event.symbol != self.symbol:
            return

        if event.side == Side.BUY:
            self._open_lots.append(
                _OpenLot(
                    entry_time=event.filled_at,
                    entry_price=float(event.price),
                    size=float(event.qty),
                    entry_fee=float(fee),
                )
            )
            return

        # SELL
        if not self._open_lots:
            return

        sell_qty_total = float(event.qty)
        if sell_qty_total <= 0:
            return

        remaining = sell_qty_total
        sell_fee_per_unit = float(fee) / sell_qty_total

        while remaining > 1e-12 and self._open_lots:
            lot = self._open_lots[0]
            close_qty = min(remaining, lot.size)

            buy_fee_part = lot.entry_fee * (close_qty / lot.size) if lot.size > 0 else 0.0
            sell_fee_part = sell_fee_per_unit * close_qty
            total_fee = buy_fee_part + sell_fee_part

            gross_pnl = (float(event.price) - lot.entry_price) * close_qty
            net_pnl = gross_pnl - total_fee

            bars_held = int((event.filled_at - lot.entry_time).total_seconds() // 60)

            self._counter += 1
            trade_id = f"{self.symbol}-T{self._counter}"

            return_pct = (float(event.price) / lot.entry_price - 1.0) * 100.0 if lot.entry_price > 0 else 0.0

            self._trades.append(
                Trade(
                    id=trade_id,
                    symbol=self.symbol,
                    side=Side.BUY,
                    entry_time=lot.entry_time,
                    exit_time=event.filled_at,
                    entry_price=lot.entry_price,
                    exit_price=float(event.price),
                    size=close_qty,
                    gross_pnl=gross_pnl,
                    fee=total_fee,
                    net_pnl=net_pnl,
                    return_pct=return_pct,
                    bars_held=bars_held,
                )
            )

            # Reduce the lot
            lot.size -= close_qty
            lot.entry_fee -= buy_fee_part
            remaining -= close_qty

            if lot.size <= 1e-12:
                self._open_lots.pop(0)

    def get_trades(self) -> List[Trade]:
        return list(self._trades)
