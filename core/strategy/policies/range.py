# core/strategy/policies/range.py
from __future__ import annotations

from dataclasses import dataclass

from core.models import Candle


"""
Intention: the dynamic strategy will map GridRangeMode + ATR settings into this mode & fields when constructing the policy. 
This keeps the policy independent from enums and Pydantic.
"""

@dataclass
class RangePolicy:
    """
    Computes the active price band (lower, upper) around the current price.

    mode:
      - "percent":  lower = close * (1 - lower_pct)
                    upper = close * (1 + upper_pct)
      - "absolute": lower = lower_price
                    upper = upper_price
      - "atr":      lower = close - ATR * atr_mult_lower
                    upper = close + ATR * atr_mult_upper

    ATR is read from candle.extra[atr_key] if mode == "atr".
    """
    mode: str  # "percent" | "absolute" | "atr"

    # percent-based range
    lower_pct: float | None = None
    upper_pct: float | None = None

    # absolute range
    lower_price: float | None = None
    upper_price: float | None = None

    # ATR-based range
    atr_mult_lower: float | None = None
    atr_mult_upper: float | None = None
    atr_key: str | None = None   # e.g. "ATR_14"

    # fallback for ATR warm-up if ATR not available
    fallback_pct: float = 0.10   # 10% range as default

    def compute(self, candle: Candle) -> tuple[float, float]:
        close = candle.close

        if self.mode == "percent":
            assert self.lower_pct is not None and self.upper_pct is not None
            lower = close * (1 - self.lower_pct)
            upper = close * (1 + self.upper_pct)

        elif self.mode == "absolute":
            assert self.lower_price is not None and self.upper_price is not None
            lower = self.lower_price
            upper = self.upper_price

        elif self.mode == "atr":
            assert self.atr_key is not None
            atr = candle.extra.get(self.atr_key)

            if atr is None:
                # Warm-up: ATR not ready -> fallback to percent range if provided,
                # otherwise use fallback_pct.
                lpct = self.lower_pct if self.lower_pct is not None else self.fallback_pct
                upct = self.upper_pct if self.upper_pct is not None else self.fallback_pct
                lower = close * (1 - lpct)
                upper = close * (1 + upct)
            else:
                mult_low = self.atr_mult_lower if self.atr_mult_lower is not None else 3.0
                mult_up = self.atr_mult_upper if self.atr_mult_upper is not None else 3.0
                lower = close - atr * mult_low
                upper = close + atr * mult_up

        else:
            raise ValueError(f"Unknown range mode {self.mode!r}")

        # Ensure lower <= upper just in case
        if lower > upper:
            lower, upper = upper, lower

        return lower, upper
