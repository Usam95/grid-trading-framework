from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, List

from pydantic import BaseModel, Field, validator

from infra.data_source import HIST_DATA_ROOT


class DataSplitConfig(BaseModel):
    """
    How to split the loaded historical data into a 'train' (backtest) segment
    and a 'forward' segment.

    This is primarily used by research/parameter-search runners.
    A simple single backtest can ignore it or just use the 'train' part.
    """

    mode: Literal["ratio", "date"] = "ratio"

    # Used when mode == "ratio"
    train_ratio: float = Field(
        0.8,
        gt=0.0,
        lt=1.0,
        description="Fraction of rows that go into the train segment (0 < r < 1).",
    )

    # Used when mode == "date"
    split_date: Optional[datetime] = Field(
        None,
        description=(
            "Calendar timestamp where the split happens when mode='date'. "
            "Rows strictly before this go to train, rows on/after go to forward."
        ),
    )

    @validator("split_date", always=True)
    def _require_split_date_for_date_mode(cls, v, values):
        mode = values.get("mode", "ratio")
        if mode == "date" and v is None:
            raise ValueError("split_date must be provided when mode='date'.")
        return v


class LocalDataConfig(BaseModel):
    """
    Configuration for loading historical candles from local files.

    Maps directly to LocalFileDataSource (or equivalent).

    YAML example:

      data:
        source: local
        symbol: "XRPUSDT"
        timeframe: "1m"
        start: "2024-11-01T00:00:00"
        end:   "2025-11-20T23:59:00"
        root: "historical_data"

        research_timeframes: ["1m", "5m", "15m"]

        split:
          mode: ratio
          train_ratio: 0.8
    """

    source: Literal["local"] = "local"

    symbol: str = Field(..., description="Symbol, e.g. XRPUSDT")

    # Target timeframe after resampling.
    # For now we assume raw data is stored at 1m resolution.
    timeframe: str = Field(
        "1m",
        description="Target timeframe, e.g. '1m', '5m', '15m', '1h', '4h', '1d'.",
    )

    # Optional explicit date range
    start: Optional[datetime] = Field(
        None,
        description="Inclusive start timestamp; if null, earliest available.",
    )
    end: Optional[datetime] = Field(
        None,
        description="Inclusive end timestamp; if null, latest available.",
    )

    root: Path = Field(
        default_factory=lambda: HIST_DATA_ROOT,
        description="Root folder where historical_data/<SYMBOL> lives.",
    )

    # Optional set of additional timeframes a research runner may want
    # to iterate over (e.g. ['1m', '5m', '15m', '1h']).
    research_timeframes: List[str] = Field(
        default_factory=list,
        description=(
            "Optional list of candidate timeframes to be used by a "
            "research/param-search runner."
        ),
    )

    # NEW: integrate the split configuration into the data config
    split: DataSplitConfig = Field(
        default_factory=DataSplitConfig,
        description=(
            "Configuration for splitting the loaded dataset into a train and "
            "forward segment. Mainly used by research runners."
        ),
    )

    class Config:
        arbitrary_types_allowed = True


# For now, we only support LocalDataConfig.
# In the future, you can introduce a Union[LocalDataConfig, BinanceLiveDataConfig, ...].
DataConfig = LocalDataConfig
