from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, Field, validator

from core.strategy.grid_strategy_simple import GridSpacing, GridRangeMode
from .strategy_base import StrategyConfigBase


class GridStrategyConfig(StrategyConfigBase):
    """
    Static grid strategy configuration.

    Mirrors core.strategy.grid_strategy_simple.GridConfig, but as a Pydantic model.

    YAML example:

      strategy:
        kind: "grid.simple"
        symbol: "XRPUSDT"
        base_order_size: 10.0
        n_levels: 10

        range_mode: "PERCENT"
        spacing: "ARITHMETIC"

        lower_pct: 0.10
        upper_pct: 0.10

        lower_price: null
        upper_price: null
    """

    kind: Literal["grid.simple"] = "grid.simple"

    symbol: str = Field(..., description="Symbol, e.g. XRPUSDT")

    base_order_size: float = Field(..., gt=0.0)
    n_levels: int = Field(
        ...,
        ge=2,
        description="Number of price levels (grid lines).",
    )

    # Percent-based range around first candle close
    lower_pct: Optional[float] = Field(
        None,
        ge=0.0,
        description="Fraction below first price, e.g. 0.1 = 10% below.",
    )
    upper_pct: Optional[float] = Field(
        None,
        ge=0.0,
        description="Fraction above first price, e.g. 0.1 = 10% above.",
    )

    # Absolute range (alternative to percent-based)
    lower_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Absolute lower bound, e.g. 0.4.",
    )
    upper_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Absolute upper bound, e.g. 0.8.",
    )

    range_mode: GridRangeMode = GridRangeMode.PERCENT
    spacing: GridSpacing = GridSpacing.ARITHMETIC

    # --- Validators -------------------------------------------------

    @validator("upper_price")
    def _check_upper_price(cls, v, values):
        """
        Only for absolute prices:
        upper_price must be > lower_price if both are set.
        """
        lower = values.get("lower_price")
        if v is not None and lower is not None and v <= lower:
            raise ValueError("upper_price must be greater than lower_price")
        return v

    # NOTE: we intentionally do NOT validate a relation between lower_pct and upper_pct,
    # because 0.10 below and 0.10 above are independent distances and
    # equality (10%/10%) is perfectly fine.

    @validator("range_mode", pre=True)
    def normalize_range_mode(cls, v):
        """
        Allow case-insensitive strings in YAML/JSON, e.g. 'PERCENT', 'percent'.
        """
        if isinstance(v, str):
            return v.lower()
        return v

    @validator("spacing", pre=True)
    def normalize_spacing(cls, v):
        """
        Allow case-insensitive strings in YAML/JSON, e.g. 'ARITHMETIC', 'arithmetic'.
        """
        if isinstance(v, str):
            return v.lower()
        return v


# For now we only have one strategy type.
# Later this can become a Union[GridStrategyConfig, OtherStrategyConfig, ...].
StrategyConfig = GridStrategyConfig
