from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .data_config import DataConfig
from .engine_config import EngineConfig, RunMode
from .strategy_grid import StrategyConfig
from .logging_config import LoggingConfig
from .research_config import GridResearchConfig


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

    # Optional: configuration for parameter search / research
    research: Optional[GridResearchConfig] = Field(
        default=None,
        description=(
            "Optional configuration for parameter-search/runs. "
            "Backtest runner can ignore this."
        ),
    )

    class Config:
        arbitrary_types_allowed = True
