# infra/config/run_config.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field("default", description="Human-readable run name.")
    description: Optional[str] = Field(None, description="Optional free-text description for this run.")
    mode: RunMode = RunMode.BACKTEST

    data: DataConfig
    engine: EngineConfig
    strategy: StrategyConfig
    logging: LoggingConfig = LoggingConfig()

    trading: Optional[TradingRuntimeConfig] = Field(
        default=None,
        description="Runtime settings for paper/live trading (execution toggles, safety, user stream).",
    )

    research: Optional[GridResearchConfig] = Field(
        default=None,
        description="Optional configuration for parameter-search/runs. Backtest runner can ignore this.",
    )

    @model_validator(mode="after")
    def _default_trading_block_for_trading_modes(self) -> "RunConfig":
        if self.mode in (RunMode.PAPER, RunMode.LIVE) and self.trading is None:
            self.trading = TradingRuntimeConfig()
        return self
