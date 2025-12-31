# infra/config/trading_config.py
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class TradingUserStreamConfig(BaseModel):
    enabled: bool = True
    keepalive_sec: int = Field(1800, ge=60, alias="keepalive_interval_sec")
    reconnect_backoff_sec: int = Field(5, ge=1)

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

class TradingStartupOrdersConfig(BaseModel):
    cancel_open_orders_on_startup: bool = True
    cancel_only_managed: bool = Field(True, alias="cancel_open_orders_only_managed")
    cancel_prefixes: List[str] = Field(default_factory=lambda: ["LT-", "GT-", "LT_"], alias="cancel_open_orders_prefixes")

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

class TradingSafetyConfig(BaseModel):
    max_open_orders: int = Field(40, ge=0)
    max_notional_per_order: float = Field(50.0, ge=0.0)
    max_position_base: float = Field(2000.0, ge=0.0)
    cancel_all_on_shutdown: bool = True

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

class TradingRuntimeConfig(BaseModel):
    """
    Runtime controls for paper/live trading.
    This replaces the old B-phase toggles.
    """
    enabled: bool = True                     # replaces execute_enabled
    require_private_endpoints: bool = True   # keep (it’s real, not a “phase”)

    user_stream: TradingUserStreamConfig = Field(default_factory=TradingUserStreamConfig)
    startup_orders: TradingStartupOrdersConfig = Field(default_factory=TradingStartupOrdersConfig)
    safety: TradingSafetyConfig = Field(default_factory=TradingSafetyConfig)

    balances_log_assets: List[str] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"