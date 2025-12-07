# core/strategy/policies/space.py
from __future__ import annotations

from dataclasses import dataclass

from core.models import Candle
from core.strategy.grid_strategy_simple import GridSpacing


@dataclass
class SpacingPolicy:
    """
    Computes grid step between neighboring price levels.

    spacing:
      - GridSpacing.ARITHMETIC
      - GridSpacing.GEOMETRIC   (reserved for future; currently treated same)

    distance_mode:
      - "percent": step = close * spacing_pct
      - "atr":     step = ATR * spacing_atr_mult

    ATR is taken from candle.extra[atr_key] if distance_mode == "atr".
    """
    spacing: GridSpacing             # ARITHMETIC / GEOMETRIC
    distance_mode: str               # "percent" | "atr"

    spacing_pct: float | None = None        # used when distance_mode="percent"
    spacing_atr_mult: float | None = None   # used when distance_mode="atr"
    atr_key: str | None = None              # e.g. "ATR_14"

    # Fallback percent spacing if ATR not yet available
    fallback_pct: float = 0.007  # 0.7%

    def compute_step(self, candle: Candle, band: tuple[float, float]) -> float:
        lower, upper = band
        close = candle.close

        if self.distance_mode == "percent":
            pct = self.spacing_pct if self.spacing_pct is not None else self.fallback_pct
            step = close * pct

        elif self.distance_mode == "atr":
            assert self.atr_key is not None, "ATR key must be set for ATR spacing"
            atr = candle.extra.get(self.atr_key)
            if atr is None:
                # Warm-up: ATR not ready -> fallback to percent
                pct = self.spacing_pct if self.spacing_pct is not None else self.fallback_pct
                step = close * pct
            else:
                mult = self.spacing_atr_mult if self.spacing_atr_mult is not None else 0.5
                step = atr * mult
        else:
            raise ValueError(f"Unknown distance mode {self.distance_mode!r}")

        # For now ARITHMETIC vs GEOMETRIC is not distinguished here.
        # GEOMETRIC spacing would adjust individual levels when building the grid.
        return step
