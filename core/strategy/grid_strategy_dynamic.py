# core/strategy/grid_strategy_dynamic.py
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
from core.strategy.policies.filter import FilterPolicy
from core.strategy.policies.range import RangePolicy
from infra.indicators import atr_key
from infra.logging_setup import get_logger


@dataclass
class DynamicGridConfig(GridConfig):
    """
    Runtime config for DynamicGridStrategy.

    It extends the basic GridConfig (percent/absolute band + spacing)
    with ATR, SL/TP, floating and filter options.

    Currently active:
      - FilterPolicy (RSI + trend)
      - ATR-based initial band via RangePolicy

    SL/TP and recentering will be wired in later.
    """

    # --- ATR / spacing / range ---
    spacing_mode: str = "percent"   # "percent" | "atr" (currently unused)
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

    # TODO (later): explicitly add atr_period_for_range if you want
    # to use something other than ATR_14.


class DynamicGridStrategy(BaseStrategy):
    """
    Dynamic / advanced grid strategy.

    Current behaviour:
      - Delegates core grid mechanics to SimpleGridStrategy.
      - Applies FilterPolicy (RSI / EMA-deviation) before calling the
        inner strategy:
          * If filters say "no": no new orders are created for this candle.
          * If filters say "yes": we lazily initialise the internal
            SimpleGridStrategy with an ATR-based band and then forward
            the candle to it.

    This matches the 'grid.dynamic' kind in DynamicGridStrategyConfig.
    """

    def __init__(self, config: DynamicGridConfig) -> None:
        self.config = config
        self.log = get_logger("strategy.grid.dynamic")

        # Will be created lazily on the first candle that passes filters
        self._inner: Optional[SimpleGridStrategy] = None

        # Filter policy (RSI + trend) uses config attributes directly
        self.filter_policy = FilterPolicy(cfg=config)

        # Range policy: we use ATR-based band as default for dynamic grid.
        # For now we assume ATR_14 is computed (see IndicatorConfig).
        # If ATR is not ready yet, RangePolicy will fall back to percent.
        fallback_pct = (
            config.lower_pct
            if config.lower_pct is not None
            else (config.upper_pct if config.upper_pct is not None else 0.10)
        )

        self.range_policy = RangePolicy(
            mode="atr",                                   # ATR-based band
            lower_pct=config.lower_pct,                  # used as fallback
            upper_pct=config.upper_pct,
            atr_mult_lower=config.range_atr_lower_mult,
            atr_mult_upper=config.range_atr_upper_mult,
            atr_key=atr_key(14),                         # ATR_14 by convention
            fallback_pct=fallback_pct,
        )

        self.activate_filters()

    # ------------------------------------------------------------------ #
    # Helper: filter logging
    # ------------------------------------------------------------------ #
    def activate_filters(self) -> None:
        filters_active: list[str] = []
        if self.config.use_rsi_filter:
            filters_active.append(
                f"RSI(period={self.config.rsi_period}, "
                f"range=[{self.config.rsi_min}, {self.config.rsi_max}]"
            )
        if self.config.use_trend_filter:
            filters_active.append(
                f"EMA(period={self.config.ema_period}, "
                f"max_dev={self.config.max_deviation_pct})"
            )

        if filters_active:
            self.log.info(
                "DynamicGridStrategy filters enabled: %s",
                "; ".join(filters_active),
            )
        else:
            self.log.info("DynamicGridStrategy filters disabled.")

    # ------------------------------------------------------------------ #
    # Helper: lazy inner initialisation with ATR-based band
    # ------------------------------------------------------------------ #
    def _ensure_inner(self, candle: Candle) -> None:
        """
        Create the internal SimpleGridStrategy (self._inner) once, based
        on the current candle and ATR-aware RangePolicy.

        If already created, this is a no-op.
        """
        if self._inner is not None:
            return

        # Compute band around current price (prefer ATR, fallback to %)
        lower, upper = self.range_policy.compute(candle)

        base_cfg = GridConfig(
            symbol=self.config.symbol,
            base_order_size=self.config.base_order_size,
            n_levels=self.config.n_levels,
            lower_price=lower,
            upper_price=upper,
            lower_pct=None,
            upper_pct=None,
            range_mode=GridRangeMode.ABSOLUTE,
            spacing=self.config.spacing,
        )
        self._inner = SimpleGridStrategy(base_cfg)

        self.log.info(
            "Initialized inner SimpleGridStrategy with band "
            "[%.6f, %.6f] (width=%.6f) at %s",
            lower,
            upper,
            upper - lower,
            candle.timestamp,
        )

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
            self.log.info(
                "Filters block trading at %s (close=%.6f)",
                candle.timestamp,
                candle.close,
            )
            return []

        # 2) Ensure internal grid is initialised once (ATR-based band)
        self._ensure_inner(candle)
        assert self._inner is not None

        # 3) Filters OK -> use normal simple grid logic
        return self._inner.on_candle(candle, account)

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        """
        Forward fills to the inner strategy. Later we may also add SL/TP
        state updates here.
        """
        if self._inner is None:
            # This should not happen: fills imply that the inner strategy
            # has already created orders.
            self.log.warning(
                "Received OrderFilledEvent but inner strategy "
                "is not initialised yet: %s",
                event,
            )
            return

        self._inner.on_order_filled(event)
