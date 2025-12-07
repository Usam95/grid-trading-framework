# core/strategy/policies/sltp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models import Candle


@dataclass
class SlTpPolicy:
    """
    Stop-loss / take-profit policy for LONG positions.

    You can choose between percent-based and ATR-based distances:

      stop_loss_type:
        - "percent": SL = entry_price * (1 - stop_loss_pct)
        - "atr":     SL = entry_price - ATR * stop_loss_atr_mult

      take_profit_type:
        - "percent": TP = entry_price * (1 + take_profit_pct)
        - "atr":     TP = entry_price + ATR * take_profit_atr_mult

    ATR is taken from candle.extra[atr_key] if needed.
    """
    # Global ATR column name to use for ATR-based SL/TP
    atr_key: Optional[str] = None

    # Stop-loss config
    use_stop_loss: bool = False
    stop_loss_type: str = "percent"         # "percent" | "atr"
    stop_loss_pct: Optional[float] = None   # e.g. 0.2 = 20%
    stop_loss_atr_mult: Optional[float] = None

    # Take-profit config
    use_take_profit: bool = False
    take_profit_type: str = "percent"       # "percent" | "atr"
    take_profit_pct: Optional[float] = None
    take_profit_atr_mult: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Compute levels at entry time (for a LONG position)
    # ------------------------------------------------------------------ #
    def compute_stop_loss(self, entry_price: float, candle: Candle) -> Optional[float]:
        if not self.use_stop_loss:
            return None

        if self.stop_loss_type == "percent":
            pct = self.stop_loss_pct
            if pct is None:
                return None
            return entry_price * (1.0 - pct)

        if self.stop_loss_type == "atr":
            if self.atr_key is None or self.stop_loss_atr_mult is None:
                return None
            atr = candle.extra.get(self.atr_key)
            if atr is None:
                # Warm-up: fallback to percent if available, otherwise no SL
                if self.stop_loss_pct is not None:
                    return entry_price * (1.0 - self.stop_loss_pct)
                return None
            return entry_price - atr * self.stop_loss_atr_mult

        raise ValueError(f"Unknown stop_loss_type {self.stop_loss_type!r}")

    def compute_take_profit(self, entry_price: float, candle: Candle) -> Optional[float]:
        if not self.use_take_profit:
            return None

        if self.take_profit_type == "percent":
            pct = self.take_profit_pct
            if pct is None:
                return None
            return entry_price * (1.0 + pct)

        if self.take_profit_type == "atr":
            if self.atr_key is None or self.take_profit_atr_mult is None:
                return None
            atr = candle.extra.get(self.atr_key)
            if atr is None:
                # Warm-up: fallback to percent if available
                if self.take_profit_pct is not None:
                    return entry_price * (1.0 + self.take_profit_pct)
                return None
            return entry_price + atr * self.take_profit_atr_mult

        raise ValueError(f"Unknown take_profit_type {self.take_profit_type!r}")

    # ------------------------------------------------------------------ #
    # Helpers for checking hits (LONG side)
    # ------------------------------------------------------------------ #
    @staticmethod
    def sl_hit_long(candle: Candle, sl: Optional[float]) -> bool:
        """
        Check if SL was hit for a LONG position within this candle.
        Conservative rule: we only look at low.
        """
        if sl is None:
            return False
        return candle.low <= sl

    @staticmethod
    def tp_hit_long(candle: Candle, tp: Optional[float]) -> bool:
        """
        Check if TP was hit for a LONG position within this candle.
        Conservative rule: we only look at high.
        """
        if tp is None:
            return False
        return candle.high >= tp
