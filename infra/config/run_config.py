from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .data_config import DataConfig
from .engine_config import EngineConfig, RunMode
from .strategy_grid import StrategyConfig
from .logging_config import LoggingConfig
from .research_config import GridResearchConfig
from .trading_config import TradingRuntimeConfig 


class RunConfig(BaseModel):
    """
    Top-level configuration for a single run.

    One YAML/JSON file → one RunConfig.

    A simple backtest only needs: data, engine, strategy, logging.

    A research runner may additionally use: data.split + research.
    """

    name: str = Field(
        "default",
        description="Human-readable run name.",
    )

    description: Optional[str] = Field(
        None,
        description="Optional free-text description for this run.",
    )

    mode: RunMode = RunMode.BACKTEST

    data: DataConfig
    engine: EngineConfig
    strategy: StrategyConfig
    logging: LoggingConfig = LoggingConfig()

    # Optional: runtime settings only used for paper/live
    trading: Optional[TradingRuntimeConfig] = Field(
        default=None,
        description="Runtime settings for paper/live trading (execution toggles, safety, user stream).",
    )

    # Optional: configuration for parameter search / research
    research: Optional[GridResearchConfig] = Field(
        default=None,
        description=(
            "Optional configuration for parameter-search/runs. "
            "Backtest runner can ignore this."
        ),
    )

    @model_validator
    def _default_trading_block_for_trading_modes(cls, values):
        mode = values.get("mode")
        if mode in (RunMode.PAPER, RunMode.LIVE) and values.get("trading") is None:
            values["trading"] = TradingRuntimeConfig()
        return values

    class Config:
        arbitrary_types_allowed = True
