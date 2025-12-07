# core/strategy/policies/recenter.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core.models import Candle


@dataclass
class RecenterPolicy:
    """
    Decides when the grid should be re-centered around the current price.

    mode:
      - "none":       never re-center
      - "band_break": re-center if price leaves a band around the grid mid
      - "time":       re-center periodically based on recenter_interval_days

    Band width logic (for band_break):
      - percent band: mid * band_pct
      - optionally ATR-based if atr_key & atr_mult are set:
          band = ATR * atr_mult
    """
    mode: str  # "none" | "band_break" | "time"

    # band-based recentering
    band_pct: float = 0.10  # 10% from mid
    atr_key: Optional[str] = None
    atr_mult: Optional[float] = None

    # time-based recentering
    recenter_interval_days: int = 30

    # internal state
    last_recenter_ts: Optional[datetime] = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Band-break logic
    # ------------------------------------------------------------------ #
    def _compute_band_width(self, candle: Candle, mid: float) -> float:
        """
        Compute band width around 'mid':
          - Prefer ATR-based if atr_key & atr_mult and ATR is available.
          - Otherwise use percent of mid.
        """
        if self.atr_key and self.atr_mult is not None:
            atr = candle.extra.get(self.atr_key)
            if atr is not None:
                return atr * self.atr_mult

        # Fallback: percentage of mid
        return mid * self.band_pct

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def should_recenter(self, candle: Candle, lower: float, upper: float) -> bool:
        """
        Decide whether we should re-center the grid given the current band
        [lower, upper] and the latest candle.

        NOTE: This does *not* mutate internal state. Call `mark_recentred`
        after you actually rebuild the grid.
        """
        if self.mode == "none":
            return False

        price = candle.close
        mid = (lower + upper) / 2.0

        if self.mode == "band_break":
            band = self._compute_band_width(candle, mid)
            upper_band = mid + band
            lower_band = mid - band
            return price > upper_band or price < lower_band

        if self.mode == "time":
            if self.last_recenter_ts is None:
                # First time: don't force recentering yet
                return False
            delta = candle.timestamp - self.last_recenter_ts
            return delta >= timedelta(days=self.recenter_interval_days)

        raise ValueError(f"Unknown recenter mode {self.mode!r}")

    def mark_recentred(self, candle: Candle) -> None:
        """
        Call this from the strategy right after you rebuild the grid.
        """
        self.last_recenter_ts = candle.timestamp
