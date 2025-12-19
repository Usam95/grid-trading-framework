#backtester\infra\config\strategy_grid.py
from __future__ import annotations

from typing import Optional, Literal, Union 

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



class DynamicGridStrategyConfig(StrategyConfigBase):
    """
    Advanced / dynamic grid strategy.

    Extends the simple grid with ATR-based spacing/range,
    floating/recenter behaviour, SL/TP and trend filters.

    YAML example:

      strategy:
        kind: "grid.dynamic"
        symbol: "XRPUSDT"
        base_order_size: 10.0
        n_levels: 15
        lower_pct: 0.10
        upper_pct: 0.10
        range_mode: "percent"
        spacing: "arithmetic"

        # --- ATR / spacing / range ---
        spacing_mode: "percent"        # or "atr"
        spacing_pct: 0.007
        spacing_atr_mult: 0.5
        range_atr_lower_mult: 3.0
        range_atr_upper_mult: 3.0

        # --- SL / TP ---
        use_stop_loss: true
        stop_loss_type: "atr"          # or "percent"
        stop_loss_pct: 0.2
        stop_loss_atr_mult: 3.0

        use_take_profit: false
        take_profit_type: "percent"    # or "atr"
        take_profit_pct: 0.3
        take_profit_atr_mult: 3.0

        # --- Floating grid / recenter ---
        recenter_mode: "band_break"    # "none" | "band_break" | "time"
        recenter_band_pct: 0.1
        recenter_atr_mult: 2.0
        recenter_interval_days: 30

        # --- Trend filters ---
        use_rsi_filter: true
        rsi_period: 14
        rsi_min: 30.0
        rsi_max: 70.0

        use_trend_filter: true
        ema_period: 100
        max_deviation_pct: 0.1
    """

    kind: Literal["grid.dynamic"] = "grid.dynamic"

    # --- core grid parameters (same as GridStrategyConfig) ---
    symbol: str
    base_order_size: float
    n_levels: int = Field(..., ge=2)

    lower_pct: Optional[float] = Field(default=None, ge=0.0)
    upper_pct: Optional[float] = Field(default=None, ge=0.0)

    lower_price: Optional[float] = Field(default=None, ge=0.0)
    upper_price: Optional[float] = Field(default=None, ge=0.0)

    range_mode: GridRangeMode = GridRangeMode.PERCENT
    spacing: GridSpacing = GridSpacing.ARITHMETIC

    # --- ATR / volatility awareness ---
    spacing_mode: Literal["percent", "atr"] = "percent"
    spacing_pct: float = Field(0.007, ge=0.0)
    spacing_atr_mult: float = Field(0.5, ge=0.0)

    range_atr_period: int = Field(14, ge=1)
    range_atr_lower_mult: float = Field(3.0, ge=0.0)
    range_atr_upper_mult: float = Field(3.0, ge=0.0)

    # --- SL / TP ---
    use_stop_loss: bool = False
    stop_loss_type: Literal["percent", "atr"] = "percent"
    stop_loss_pct: float = Field(0.2, ge=0.0)
    stop_loss_atr_mult: float = Field(3.0, ge=0.0)

    use_take_profit: bool = False
    take_profit_type: Literal["percent", "atr"] = "percent"
    take_profit_pct: float = Field(0.3, ge=0.0)
    take_profit_atr_mult: float = Field(3.0, ge=0.0)
    # NEW: which ATR period to use for SL/TP (must be present in indicators.atr_periods)
    
    sltp_atr_period: int = Field(14, ge=1)


    # --- Floating / recentering ---
    class FloatingGridConfig(BaseModel):
        enabled: bool = False
        mode: Literal["band_break", "time"] = "band_break"
        band_pct: float = Field(0.10, ge=0.0)

        # If true: band width uses ATR * atr_mult, otherwise uses mid * band_pct
        use_atr_band: bool = True
        atr_period: Optional[int] = Field(
            None, ge=1,
            description="ATR period for band sizing; if null, reuse range_atr_period."
        )
        atr_mult: float = Field(2.0, ge=0.0)

        # used only when mode == "time"
        interval_days: int = Field(30, ge=1)

    floating_grid: FloatingGridConfig = Field(default_factory=FloatingGridConfig)


    # --- Trend filters ---
    use_rsi_filter: bool = False
    rsi_period: int = Field(14, ge=1)
    rsi_min: float = Field(30.0, ge=0.0, le=100.0)
    rsi_max: float = Field(70.0, ge=0.0, le=100.0)

    use_trend_filter: bool = False
    ema_period: int = Field(100, ge=1)
    max_deviation_pct: float = Field(0.1, ge=0.0)


    @validator("range_mode", pre=True)
    def normalize_range_mode(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @validator("spacing", pre=True)
    def normalize_spacing(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    # (Optional) later we can add validators to ensure sensible combinations.


#StrategyConfig = GridStrategyConfig
# Now we support both the simple and dynamic grid configs.
StrategyConfig = Union[GridStrategyConfig, DynamicGridStrategyConfig]
