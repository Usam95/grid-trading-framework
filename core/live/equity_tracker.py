# core/live/equity_tracker.py
from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Optional, Tuple

from core.models import AccountState


@dataclass
class Balance:
    free: float
    locked: float


class EquityTracker:
    def __init__(self, *, symbol: str, exchange: Any) -> None:
        self.symbol = symbol.upper()
        self.exchange = exchange
        self.base, self.quote = self._split_symbol(self.symbol)
        self.last_price: float = 0.0

    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        # For now: assume ...USDT. Later: use exchangeInfo baseAsset/quoteAsset.
        if symbol.endswith("USDT"):
            return symbol[:-4], "USDT"
        raise ValueError(f"Unsupported symbol split for {symbol}")

    def snapshot(self, *, last_price: Optional[float] = None) -> tuple[float, Balance, Balance]:
        """
        Snapshot balances + price.

        IMPORTANT:
        - If last_price is provided, we use it (no REST ticker call).
        - Else if we already have self.last_price cached, we use it.
        - Else we fetch ticker once.
        """
        # price selection
        if last_price is not None and float(last_price) > 0.0:
            px = float(last_price)
            self.last_price = px
        elif self.last_price > 0.0:
            px = float(self.last_price)
        else:
            t = self.exchange.client.get_symbol_ticker(symbol=self.symbol)
            px = float(t["price"])
            self.last_price = px

        # balances
        bals = self.exchange.get_balances(non_zero_only=False)
        b_free, b_locked = bals.get(self.base, (0.0, 0.0))
        q_free, q_locked = bals.get(self.quote, (0.0, 0.0))

        return px, Balance(float(b_free), float(b_locked)), Balance(float(q_free), float(q_locked))

    # ---------------------------------------------------------------------
    # Compatibility API
    # ---------------------------------------------------------------------
    def refresh(self, *, last_price: float) -> None:
        """
        Called by runtime on each candle close. Just cache the price.
        """
        self.last_price = float(last_price)

    def snapshot_json(self) -> dict[str, Any]:
        px, b, q = self.snapshot()
        return {
            "symbol": self.symbol,
            "price": float(px),
            "base": asdict(b) if is_dataclass(b) else b,
            "quote": asdict(q) if is_dataclass(q) else q,
        }

    # ---------------------------------------------------------------------
    # Strategy boundary convenience (optional)
    # ---------------------------------------------------------------------
    def account_state(self) -> AccountState:
        bals = self.exchange.get_balances(non_zero_only=False)
        b_free, b_locked = bals.get(self.base, (0.0, 0.0))
        q_free, q_locked = bals.get(self.quote, (0.0, 0.0))

        px = float(self.last_price)
        if px <= 0.0:
            t = self.exchange.client.get_symbol_ticker(symbol=self.symbol)
            px = float(t["price"])
            self.last_price = px

        cash_balance = float(q_free + q_locked)
        total_value = cash_balance + float(b_free + b_locked) * px

        return AccountState(
            cash_balance=cash_balance,
            total_value=total_value,
            invested_cost=0.0,  # until you track cost basis via fills
            base_free=float(b_free),
            base_locked=float(b_locked),
            quote_free=float(q_free),
            quote_locked=float(q_locked),
        )
