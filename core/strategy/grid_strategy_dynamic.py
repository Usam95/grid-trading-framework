# core/strategy/grid_strategy_dynamic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.models import Candle, AccountState, Order, OrderFilledEvent
from core.strategy.base import BaseStrategy
from core.strategy.grid_strategy_simple import (
    GridConfig,
    GridRangeMode,
    GridSpacing,
    SimpleGridStrategy,
)
from core.strategy.policies.filter import FilterPolicy
from infra.logging_setup import get_logger


@dataclass
class DynamicGridConfig(GridConfig):
    """
    Runtime config for DynamicGridStrategy.

    It extends the basic GridConfig (percent/absolute band + spacing)
    with ATR, SL/TP, floating and filter options.

    For now, ATR / SLTP / recentering are not yet wired in. Only the
    filter policy (RSI + trend) is active. In later steps we will
    gradually activate the remaining policies.
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

    Current behaviour:
      - Delegates core grid mechanics to SimpleGridStrategy.
      - Applies FilterPolicy (RSI / EMA-deviation) before calling the
        inner strategy:
          * If filters say "no": no new orders are created for this candle.
          * If filters say "yes": we forward the candle to the inner
            SimpleGridStrategy and return its orders.

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

        # Build filter policy (RSI + trend). It expects a config-like
        # object with attributes:
        #   use_rsi_filter, rsi_period, rsi_min, rsi_max,
        #   use_trend_filter, ema_period, max_deviation_pct
        self.filter_policy = FilterPolicy(cfg=config)

        filters_active: list[str] = []
        if config.use_rsi_filter:
            filters_active.append(
                f"RSI(period={config.rsi_period}, "
                f"range=[{config.rsi_min}, {config.rsi_max}]"
            )
        if config.use_trend_filter:
            filters_active.append(
                f"EMA(period={config.ema_period}, "
                f"max_dev={config.max_deviation_pct})"
            )

        if filters_active:
            self.log.info("DynamicGridStrategy filters enabled: %s", "; ".join(filters_active))
        else:
            self.log.info("DynamicGridStrategy filters disabled.")

    # ------------------------------------------------------------------ #
    # BaseStrategy interface
    # ------------------------------------------------------------------ #
    def on_candle(self, candle: Candle, account: AccountState) -> List[Order]:
        """
        Apply filters first; only if they allow trading, delegate to
        the underlying SimpleGridStrategy.

        IMPORTANT: When filters block trading, we DO NOT call the inner
        strategy at all. This way, the inner grid does not build or
        advance its state until the environment is allowed again.
        """
        # 1) Check filters (RSI / trend)
        if not self.filter_policy.allow_trading(candle):
            # No new orders when filters are violated.
            self.log.info(
                "Filters block trading at %s (close=%.6f)",
                candle.timestamp,
                candle.close,
            )
            return []

        # 2) Filters OK -> use normal simple grid logic
        return self._inner.on_candle(candle, account)

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        """
        Forward fills to the inner strategy. Later we may also add SL/TP
        state updates here.
        """
        self._inner.on_order_filled(event)
