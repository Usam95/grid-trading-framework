# infra/config_models.py
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, Field, validator

from infra.data_source import HIST_DATA_ROOT
from core.strategy.grid_strategy_simple import GridSpacing, GridRangeMode


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RunMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


# ---------------------------------------------------------------------------
# Data config (where candles come from)
# ---------------------------------------------------------------------------

class LocalDataConfig(BaseModel):
    """
    Configuration for loading historical candles from local files.

    Maps directly to LocalFileDataSource.
    """
    source: Literal["local"] = "local"

    symbol: str = Field(..., description="Symbol, e.g. XRPUSDT")
    start: Optional[datetime] = Field(
        None,
        description="Inclusive start timestamp; if null, earliest available."
    )
    end: Optional[datetime] = Field(
        None,
        description="Inclusive end timestamp; if null, latest available."
    )

    root: Path = Field(
        default_factory=lambda: HIST_DATA_ROOT,
        description="Root folder where historical_data/<SYMBOL> lives."
    )

    class Config:
        arbitrary_types_allowed = True


# In the future, you can add:
# class BinanceLiveDataConfig(BaseModel):
#     source: Literal["binance_live"] = "binance_live"
#     symbol: str
#     api_key: str
#     api_secret: str
#     ...


DataConfig = LocalDataConfig  # for now, we only support local; keep alias for future Union


# ---------------------------------------------------------------------------
# Engine config (backtest / live engine behaviour)
# ---------------------------------------------------------------------------

class BacktestEngineConfig(BaseModel):
    """
    Generic config for the backtest engine.

    This will later be extended (slippage model, fill model, etc.).
    """
    mode: RunMode = RunMode.BACKTEST

    initial_balance: float = Field(
        ...,
        gt=0,
        description="Starting cash in quote currency, e.g. 1000.0 USDT."
    )
    trading_fee_pct: float = Field(
        0.0,
        ge=0.0,
        description="Fee percentage per side, e.g. 0.0007 = 0.07%."
    )
    slippage_pct: float = Field(
        0.0,
        ge=0.0,
        description="Optional slippage model (% of price) for fills."
    )

    # optional limits for safety/faster tests
    max_candles: Optional[int] = Field(
        None,
        description="Optionally limit number of candles processed for this run."
    )


EngineConfig = BacktestEngineConfig  # alias for future live engine configs


# ---------------------------------------------------------------------------
# Strategy config (grid, later: others)
# ---------------------------------------------------------------------------

class GridStrategyConfig(BaseModel):
    """
    Static grid strategy configuration (matches core.strategy.grid_strategy_simple.GridConfig).
    """
    kind: Literal["grid.simple"] = "grid.simple"

    symbol: str = Field(..., description="Symbol, e.g. XRPUSDT")

    base_order_size: float = Field(..., gt=0.0)
    n_levels: int = Field(..., ge=2, description="Number of price levels (grid lines).")

    # Percent-based range around first candle close
    lower_pct: Optional[float] = Field(
        None,
        ge=0.0,
        description="Fraction below first price, e.g. 0.1 = 10% below."
    )
    upper_pct: Optional[float] = Field(
        None,
        ge=0.0,
        description="Fraction above first price, e.g. 0.1 = 10% above."
    )

    # Absolute range (alternative to percent-based)
    lower_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Absolute lower bound, e.g. 0.4."
    )
    upper_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Absolute upper bound, e.g. 0.8."
    )

    range_mode: GridRangeMode = GridRangeMode.PERCENT
    spacing: GridSpacing = GridSpacing.ARITHMETIC

    # --- Validators -------------------------------------------------

    @validator("upper_price")
    def _check_upper_price(cls, v, values):
        """
        Keep this check ONLY for absolute prices:
        upper_price must be > lower_price if both are set.
        """
        lower = values.get("lower_price")
        if v is not None and lower is not None and v <= lower:
            raise ValueError("upper_price must be greater than lower_price")
        return v

    # NOTE: we REMOVE the upper_pct / lower_pct relation validator entirely,
    # because 0.10 below and 0.10 above are independent distances.
    # So lower_pct == upper_pct is perfectly fine.

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


StrategyConfig = GridStrategyConfig  # alias for future multi-strategy setups


# ---------------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------------

class LoggingConfig(BaseModel):
    """
    Controls basic logging behaviour.

    Interacts with infra.logging_setup.get_logger().
    """
    name: str = Field(
        "gridbt",
        description="Logger name; used as root for module loggers."
    )
    level: str = Field(
        "INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR."
    )
    log_dir: str = Field(
        "logs",
        description="Directory for log files."
    )
    to_console: bool = True
    to_file: bool = True


# ---------------------------------------------------------------------------
# Top-level run config
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    """
    Top-level configuration for a single run.

    One YAML/JSON file → one RunConfig → one backtest (for now).
    Later you can add support for multi-symbol runs, portfolios, etc.
    """
    name: str = Field("default", description="Human-readable run name.")
    description: Optional[str] = Field(
        None,
        description="Optional free-text description for this run."
    )

    mode: RunMode = RunMode.BACKTEST

    data: DataConfig
    engine: EngineConfig
    strategy: StrategyConfig
    logging: LoggingConfig = LoggingConfig()
