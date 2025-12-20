# infra/config/binance_live_data_config.py
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BinanceLiveDataConfig(BaseModel):
    """
    Live market-data configuration from Binance WebSocket kline streams.

    YAML example:

      data:
        source: binance
        symbol: "XRPUSDT"
        timeframe: "1m"
        use_testnet_ws: true
    """

    source: Literal["binance"] = "binance"

    symbol: str = Field(..., description="Symbol, e.g. XRPUSDT")
    timeframe: str = Field("1m", description="Binance kline interval, e.g. '1m', '5m', '1h'")

    # If None, we infer from RunConfig.mode (PAPER => testnet ws)
    use_testnet_ws: Optional[bool] = Field(
        default=None,
        description="If true, uses Binance Spot Testnet websocket stream endpoints.",
    )

    # Optional override. If not provided, we pick defaults for prod/testnet.
    ws_base_url: Optional[str] = Field(
        default=None,
        description="Optional websocket base url override (rarely needed).",
    )
