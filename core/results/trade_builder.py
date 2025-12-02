# core/results/trade_builder.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from core.models import OrderFilledEvent, Side
from core.results.models import Trade


@dataclass
class _OpenLot:
    size: float
    entry_price: float
    entry_time: datetime
    fees: float  # accumulated buy-side fees


class TradeBuilder:
    """
    MVP TradeBuilder: long-only, FIFO, no partial closes.

    - On BUY fill: open a new lot.
    - On SELL fill: close the oldest lot and create a Trade.

    This mirrors the current engine behaviour:
      * each BUY creates a separate position
      * each SELL closes exactly one position (same size)
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._open_lots: List[_OpenLot] = []
        self._trades: List[Trade] = []
        self._counter: int = 0

    def on_fill(self, event: OrderFilledEvent, fee: float) -> None:
        if event.symbol != self.symbol:
            return

        if event.side == Side.BUY:
            # Open a new lot
            self._open_lots.append(
                _OpenLot(
                    size=event.qty,
                    entry_price=event.price,
                    entry_time=event.filled_at,
                    fees=fee,
                )
            )

        elif event.side == Side.SELL:
            if not self._open_lots:
                # Nothing to close (shouldn't happen in grid spot trading)
                return

            # FIFO: close the earliest lot
            lot = self._open_lots.pop(0)

            # MVP assumption: qty matches lot.size (no partial closes)
            qty = event.qty

            # PnL components
            gross_pnl = (event.price - lot.entry_price) * qty
            total_fees = lot.fees + fee
            net_pnl = gross_pnl - total_fees

            # Rough bars_held in minutes (since we use 1m candles)
            bars_held = int(
                (event.filled_at - lot.entry_time).total_seconds() // 60
            )

            # Informative ID like "XRPUSDT-T1"
            self._counter += 1
            trade_id = f"{self.symbol}-T{self._counter}"

            trade = Trade(
                # 👇 this is the actual field name that exists in the dataclass
                id=trade_id,

                symbol=self.symbol,
                side=Side.BUY,  # long trade

                entry_time=lot.entry_time,
                exit_time=event.filled_at,
                entry_price=lot.entry_price,
                exit_price=event.price,
                size=qty,

                gross_pnl=gross_pnl,
                fee=total_fees,
                net_pnl=net_pnl,
                return_pct=(
                    (net_pnl / (lot.entry_price * qty)) * 100.0
                    if lot.entry_price * qty > 0
                    else 0.0
                ),
                bars_held=bars_held,
            )
            self._trades.append(trade)

    def get_trades(self) -> List[Trade]:
        return list(self._trades)
