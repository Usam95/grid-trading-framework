# core/strategy/policies/sltp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models import Candle
from infra.indicators import atr_key


@dataclass
class SLTPPolicy:
    """
    Global stop-loss / take-profit policy for DynamicGridStrategy.

    It works like this:
      - On the first candle, it remembers a base_price (current close).
      - It computes stop_loss_price / take_profit_price using either:
          * percent: base_price * (1 ± pct)
          * ATR:     base_price ± atr_mult * ATR(period)
      - On each subsequent candle, it checks:
          * if Low <= stop_loss_price -> "stop_loss"
          * if High >= take_profit_price -> "take_profit"

    The 'cfg' object is expected to have attributes:
      use_stop_loss, stop_loss_type, stop_loss_pct, stop_loss_atr_mult,
      use_take_profit, take_profit_type, take_profit_pct, take_profit_atr_mult,
      sltp_atr_period (int)
    """

    cfg: object                  # DynamicGridConfig or compatible
    sltp_atr_period: int = 14    # ATR period to use for SL/TP

    base_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Reset internal state to allow re-initialisation (for re-entry later).
        """
        self.base_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

    def _init_levels_if_needed(self, candle: Candle) -> None:
        if self.base_price is not None:
            return

        # Base reference price - simplest: current close
        self.base_price = candle.close

        # Fetch ATR if any of SL/TP uses ATR
        atr_val = None
        if (
            getattr(self.cfg, "stop_loss_type", None) == "atr"
            or getattr(self.cfg, "take_profit_type", None) == "atr"
        ):
            key = atr_key(self.sltp_atr_period)
            atr_val = candle.extra.get(key)

        # --- STOP LOSS ---
        if getattr(self.cfg, "use_stop_loss", False):
            sl_type = getattr(self.cfg, "stop_loss_type", "percent")
            if sl_type == "percent":
                pct = getattr(self.cfg, "stop_loss_pct", 0.0)
                self.stop_loss_price = self.base_price * (1.0 - pct)
            elif sl_type == "atr" and atr_val is not None:
                mult = getattr(self.cfg, "stop_loss_atr_mult", 0.0)
                self.stop_loss_price = self.base_price - mult * atr_val

        # --- TAKE PROFIT ---
        if getattr(self.cfg, "use_take_profit", False):
            tp_type = getattr(self.cfg, "take_profit_type", "percent")
            if tp_type == "percent":
                pct = getattr(self.cfg, "take_profit_pct", 0.0)
                self.take_profit_price = self.base_price * (1.0 + pct)
            elif tp_type == "atr" and atr_val is not None:
                mult = getattr(self.cfg, "take_profit_atr_mult", 0.0)
                self.take_profit_price = self.base_price + mult * atr_val

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check(self, candle: Candle) -> Optional[str]:
        """
        Returns:
          - "stop_loss"   if SL is triggered
          - "take_profit" if TP is triggered
          - None          otherwise
        """
        use_sl = getattr(self.cfg, "use_stop_loss", False)
        use_tp = getattr(self.cfg, "use_take_profit", False)
        if not (use_sl or use_tp):
            return None

        self._init_levels_if_needed(candle)

        # Stop-loss: use Low
        if use_sl and self.stop_loss_price is not None:
            if candle.low <= self.stop_loss_price:
                return "stop_loss"

        # Take-profit: use High
        if use_tp and self.take_profit_price is not None:
            if candle.high >= self.take_profit_price:
                return "take_profit"

        return None
