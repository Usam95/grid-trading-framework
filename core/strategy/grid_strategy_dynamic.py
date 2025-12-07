from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.models import Candle, AccountState, Order, OrderFilledEvent
from core.strategy.base import BaseStrategy
from core.strategy.grid_strategy_simple import (
    GridConfig,
    GridRangeMode,
    GridSpacing,
    SimpleGridStrategy,
)
from infra.logging_setup import get_logger


@dataclass
class DynamicGridConfig(GridConfig):
    """
    Runtime config for DynamicGridStrategy.

    It extends the basic GridConfig (percent/absolute band + spacing)
    with ATR, SL/TP, floating and filter options.

    For now, these extra fields are not yet used – the strategy behaves
    exactly like SimpleGridStrategy. In later steps we will gradually
    activate them.
    """

    # --- ATR / spacing / range ---
    spacing_mode: str = "percent"   # "percent" | "atr"
    spacing_pct: float = 0.007
    spacing_atr_mult: float = 0.5
    range_atr_lower_mult: float = 3.0
    range_atr_upper_mult: float = 3.0

    # --- SL / TP ---
    use_stop_loss: bool = False
    stop_loss_type: str = "percent"  # "percent" | "atr"
    stop_loss_pct: float = 0.2
    stop_loss_atr_mult: float = 3.0

    use_take_profit: bool = False
    take_profit_type: str = "percent"  # "percent" | "atr"
    take_profit_pct: float = 0.3
    take_profit_atr_mult: float = 3.0

    # --- Floating / recentering ---
    recenter_mode: str = "none"  # "none" | "band_break" | "time"
    recenter_band_pct: float = 0.1
    recenter_atr_mult: float = 2.0
    recenter_interval_days: int = 30

    # --- Trend filters ---
    use_rsi_filter: bool = False
    rsi_period: int = 14
    rsi_min: float = 30.0
    rsi_max: float = 70.0

    use_trend_filter: bool = False
    ema_period: int = 100
    max_deviation_pct: float = 0.1


class DynamicGridStrategy(BaseStrategy):
    """
    Dynamic / advanced grid strategy.

    v0 implementation: thin wrapper around SimpleGridStrategy so that
    we can wire the config and backtest pipeline without changing
    behaviour yet. In next steps we'll start using the extra fields
    (ATR, floating grid, filters, SL/TP).

    This matches the 'grid.dynamic' kind in DynamicGridStrategyConfig.
    """

    def __init__(self, config: DynamicGridConfig) -> None:
        self.config = config
        self.log = get_logger("strategy.grid.dynamic")

        # For now, delegate core grid behaviour to SimpleGridStrategy.
        base_cfg = GridConfig(
            symbol=config.symbol,
            base_order_size=config.base_order_size,
            n_levels=config.n_levels,
            lower_pct=config.lower_pct,
            upper_pct=config.upper_pct,
            lower_price=config.lower_price,
            upper_price=config.upper_price,
            range_mode=config.range_mode,
            spacing=config.spacing,
        )
        self._inner = SimpleGridStrategy(base_cfg)

    def on_candle(self, candle: Candle, account: AccountState) -> List[Order]:
        # Later we’ll intercept here: apply filters, ATR-based range,
        # recenter logic, SL/TP checks, etc.
        return self._inner.on_candle(candle, account)

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        # Later we may also track SL/TP or recenter state here.
        self._inner.on_order_filled(event)
