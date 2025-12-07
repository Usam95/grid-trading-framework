# core/strategy/policies/filter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.models import Candle
from infra.indicators import rsi_key, ema_key


class FilterCfgLike(Protocol):
    """
    Structural type for anything that can configure the FilterPolicy.

    Expected attributes (your DynamicGridConfig / Pydantic config
    should provide the same names):

      use_rsi_filter: bool
      rsi_period: int
      rsi_min: float
      rsi_max: float

      use_trend_filter: bool
      ema_period: int          # <- matches DynamicGridStrategyConfig
      max_deviation_pct: float
    """
    use_rsi_filter: bool
    rsi_period: int
    rsi_min: float
    rsi_max: float

    use_trend_filter: bool
    ema_period: int
    max_deviation_pct: float


@dataclass
class FilterPolicy:
    """
    Combines RSI and trend (EMA deviation) filters.

    - If RSI is outside [rsi_min, rsi_max] -> block new trades.
    - If price deviates from EMA by more than max_deviation_pct -> block new trades.
    """
    cfg: FilterCfgLike

    # ----- RSI filter -----------------------------------------------------
    def _allow_rsi(self, candle: Candle) -> bool:
        if not getattr(self.cfg, "use_rsi_filter", False):
            return True

        key = rsi_key(self.cfg.rsi_period)
        rsi = candle.extra.get(key)

        # Warm-up phase: indicator not ready yet -> do not block trading
        if rsi is None:
            return True

        return self.cfg.rsi_min <= rsi <= self.cfg.rsi_max

    # ----- Trend filter (EMA deviation) ----------------------------------
    def _allow_trend(self, candle: Candle) -> bool:
        if not getattr(self.cfg, "use_trend_filter", False):
            return True

        key = ema_key(self.cfg.ema_period)
        ema = candle.extra.get(key)

        # Warm-up: EMA not ready yet -> do not block
        if ema is None:
            return True

        dev = abs(candle.close - ema) / ema
        return dev <= self.cfg.max_deviation_pct

    # ----- Combined decision ---------------------------------------------
    def allow_trading(self, candle: Candle) -> bool:
        """
        Return True if all active filters allow opening new trades for this candle.
        """
        return self._allow_rsi(candle) and self._allow_trend(candle)
