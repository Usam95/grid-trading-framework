from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class RunMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class BacktestEngineConfig(BaseModel):
    """
    Generic config for the backtest engine.

    This will later be extended (slippage model, fill model, etc.).
    """

    mode: RunMode = RunMode.BACKTEST

    track_equity_curve: bool = Field(
        True,
        description="If True, keep full equity curve in memory.",
    )

    initial_balance: float = Field(
        ...,
        gt=0,
        description="Starting cash in quote currency, e.g. 1000.0 USDT.",
    )

    trading_fee_pct: float = Field(
        0.0,
        ge=0.0,
        description="Fee percentage per side, e.g. 0.0007 = 0.07%.",
    )
    slippage_pct: float = Field(
        0.0,
        ge=0.0,
        description="Optional slippage model (% of price) for fills.",
    )

    # Optional limits for safety/faster tests
    max_candles: Optional[int] = Field(
        None,
        description="Optionally limit number of candles processed for this run.",
    )

    # Which metrics to compute via MetricRegistry
    metrics: List[str] = Field(
        default_factory=lambda: [
            "net_pnl",
            "total_return_pct",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
        ],
        description=(
            "List of metric names to compute via the MetricRegistry. "
            "Names must be registered in "
            "core.results.metrics.create_default_metric_registry()."
        ),
    )


# Alias for future live engine configs
EngineConfig = BacktestEngineConfig
