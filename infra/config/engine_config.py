# infra/config/engine_config.py
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class RunMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class InsufficientFundsMode(str, Enum):
    SKIP = "skip"
    RESIZE = "resize"


class BootstrapMode(str, Enum):
    LONG_ONLY = "long_only"
    NEUTRAL_SPLIT = "neutral_split"
    NEUTRAL_TOPUP = "neutral_topup"


class ConstraintConfig(BaseModel):
    """
    Engine-level feasibility checks for PLACE_ORDER actions.
    """
    enabled: bool = True
    insufficient_funds_mode: InsufficientFundsMode = InsufficientFundsMode.SKIP
    min_order_qty: float = Field(
        0.0,
        ge=0.0,
        description="Minimum base-asset quantity. Orders below are skipped.",
    )

    @validator("insufficient_funds_mode", pre=True)
    def _norm_mode(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


class ReservationConfig(BaseModel):
    """
    Spot-like reservation: lock quote for BUY orders and base qty for SELL orders
    when orders are accepted (until filled or cancelled).
    """
    enabled: bool = True


class BootstrapConfig(BaseModel):
    """
    Portfolio bootstrap before the first candle is processed.

    - long_only: do nothing (you start with quote cash only, unless initial_base_qty > 0)
    - neutral_split: convert a percentage of quote cash into base at the first candle close
    - neutral_topup: ensure base holdings reach a target (qty or value% of equity)
    """
    mode: BootstrapMode = BootstrapMode.LONG_ONLY

    # neutral_split
    initial_quote_to_base_pct: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="For neutral_split: fraction of initial quote cash to convert into base at start.",
    )

    # neutral_topup
    target_base_qty: Optional[float] = Field(None, ge=0.0)
    target_base_value_pct: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_topup_quote: Optional[float] = Field(None, ge=0.0)

    @validator("mode", pre=True)
    def _norm_bootstrap_mode(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


class BacktestEngineConfig(BaseModel):
    """
    Configuration for BacktestEngine.

    Notes:
    - initial_balance is quote-currency cash (e.g. USDT/EUR depending on the symbol).
    - initial_base_qty lets you start with some already-owned base asset for the symbol.
    """
    mode: RunMode = RunMode.BACKTEST

    initial_balance: float = Field(1000.0, ge=0.0)
    initial_base_qty: float = Field(0.0, ge=0.0)

    trading_fee_pct: float = Field(0.001, ge=0.0)
    slippage_pct: float = Field(0.0, ge=0.0)

    max_candles: Optional[int] = Field(None, ge=1)

    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    reservations: ReservationConfig = Field(default_factory=ReservationConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    metrics: List[str] = Field(default_factory=list)

    @validator("mode", pre=True)
    def _norm_run_mode(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


EngineConfig = BacktestEngineConfig
