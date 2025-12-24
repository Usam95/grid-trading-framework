# core/live/equity_tracker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
from dataclasses import asdict, is_dataclass
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

    def snapshot(self) -> tuple[float, Balance, Balance]:
        # price
        t = self.exchange.client.get_symbol_ticker(symbol=self.symbol)
        px = float(t["price"])

        bals = self.exchange.get_balances(non_zero_only=False)
        b_free, b_locked = bals.get(self.base, (0.0, 0.0))
        q_free, q_locked = bals.get(self.quote, (0.0, 0.0))
        
        # keep last price for account_state() consumers (optional convenience)
        self.last_price = px
        
        return px, Balance(b_free, b_locked), Balance(q_free, q_locked)

    # ---------------------------------------------------------------------
    # Compatibility API (live_trade.py expects this)
    # ---------------------------------------------------------------------
    def refresh(self, *, last_price: float) -> None:
        """
        Compatibility shim.
        live_trade.py calls equity.refresh(last_price=...).
        Internally forward to the existing method you already have.
        """
        # If your class already has one of these, call it:
        if hasattr(self, "update") and callable(getattr(self, "update")):
            self.update(last_price=last_price)  # type: ignore[attr-defined]
            return
        if hasattr(self, "on_price") and callable(getattr(self, "on_price")):
            self.on_price(last_price=last_price)  # type: ignore[attr-defined]
            return
        if hasattr(self, "tick") and callable(getattr(self, "tick")):
            self.tick(last_price=last_price)  # type: ignore[attr-defined]
            return

        # Otherwise, implement minimal behavior:
        self.last_price = float(last_price)  # requires you to have self.last_price

    def snapshot_json(self) -> dict[str, Any]:
        """
        Return a JSON-serializable snapshot (no Balance objects).
        """
        s = self.snapshot()  # your existing snapshot method
        # Convert dataclasses like Balance to dicts
        def conv(x: Any) -> Any:
            if is_dataclass(x):
                return asdict(x)
            return x
        if isinstance(s, dict):
            return {k: conv(v) for k, v in s.items()}
        return {"snapshot": conv(s)}
    
    # ---------------------------------------------------------------------
    # Required by live_trade.py: provide AccountState for strategy boundary
    # ---------------------------------------------------------------------
    def account_state(self) -> AccountState:
        bals = self.exchange.get_balances(non_zero_only=False)
        b_free, b_locked = bals.get(self.base, (0.0, 0.0))
        q_free, q_locked = bals.get(self.quote, (0.0, 0.0))

        # choose a price: use cached last_price if available, otherwise fetch
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
            invested_cost=0.0,         # until you track cost basis via fills
            base_free=float(b_free),
            base_locked=float(b_locked),
            quote_free=float(q_free),
            quote_locked=float(q_locked),
        )
