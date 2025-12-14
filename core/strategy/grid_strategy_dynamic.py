# core/strategy/grid_strategy_dynamic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.engine_actions import EngineAction
from core.models import Candle, AccountState, OrderFilledEvent
from core.strategy.base import BaseStrategy
from core.strategy.grid_strategy_simple import (
    GridConfig,
    GridRangeMode,
    GridSpacing,
    SimpleGridStrategy,
)

from core.strategy.policies.sltp import SLTPPolicy
from core.strategy.policies.filter import FilterPolicy
from core.strategy.policies.range import RangePolicy
from infra.indicators import atr_key
from infra.logging_setup import get_logger


@dataclass
class DynamicGridConfig(GridConfig):
    """
    Runtime config for DynamicGridStrategy.

    Matches infra.config.DynamicGridStrategyConfig fields.
    """

    # --- ATR / spacing / range ---
    spacing_mode: str = "percent"   # "percent" | "atr" (currently unused)
    spacing_pct: float = 0.007
    spacing_atr_mult: float = 0.5

    range_atr_period: int = 50
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

    sltp_atr_period: int = 14

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

    FIXES in this version:
      - returns List[EngineAction] (consistent with engine interface)
      - global SL/TP now returns EngineAction.grid_exit(...)
      - inner grid is still SimpleGridStrategy (returns EngineActions)
    """

    def __init__(self, config: DynamicGridConfig) -> None:
        self.config = config
        self.log = get_logger("strategy.grid.dynamic")

        self._inner: Optional[SimpleGridStrategy] = None
        self._stopped: bool = False

        self.filter_policy = FilterPolicy(cfg=config)

        fallback_pct = (
            config.lower_pct
            if config.lower_pct is not None
            else (config.upper_pct if config.upper_pct is not None else 0.10)
        )

        self.range_policy = RangePolicy(
            mode="atr",
            lower_pct=config.lower_pct,
            upper_pct=config.upper_pct,
            atr_mult_lower=config.range_atr_lower_mult,
            atr_mult_upper=config.range_atr_upper_mult,
            atr_key=atr_key(config.range_atr_period),
            fallback_pct=fallback_pct,
        )

        self.sltp_policy = SLTPPolicy(cfg=config, sltp_atr_period=config.sltp_atr_period)

        self._log_filters()

    def _log_filters(self) -> None:
        filters_active: list[str] = []
        if self.config.use_rsi_filter:
            filters_active.append(
                f"RSI(period={self.config.rsi_period}, range=[{self.config.rsi_min}, {self.config.rsi_max}])"
            )
        if self.config.use_trend_filter:
            filters_active.append(
                f"EMA(period={self.config.ema_period}, max_dev={self.config.max_deviation_pct})"
            )

        if filters_active:
            self.log.info("DynamicGridStrategy filters enabled: %s", "; ".join(filters_active))
        else:
            self.log.info("DynamicGridStrategy filters disabled.")

    def _ensure_inner(self, candle: Candle) -> None:
        if self._inner is not None:
            return

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
            "Initialized inner SimpleGridStrategy with band [%.6f, %.6f] at %s",
            lower,
            upper,
            candle.timestamp,
        )

    def on_candle(self, candle: Candle, account: AccountState) -> List[EngineAction]:
        """
        Order of operations:

          1. If already stopped -> no more actions.
          2. Check global SL/TP:
               - if triggered -> emit GRID_EXIT action.
          3. Apply filters (RSI + trend).
          4. Ensure inner grid exists (ATR band).
          5. Delegate to SimpleGridStrategy.
        """
        if self._stopped:
            return []

        # Global SL/TP check
        reason = self.sltp_policy.check(candle)
        if reason is not None:
            self.log.info(
                "DynamicGridStrategy global %s triggered at %s -> requesting engine flatten & cancel.",
                reason,
                candle.timestamp,
            )
            self._stopped = True
            self.sltp_policy.reset()
            return [EngineAction.grid_exit(symbol=self.config.symbol, exit_reason=reason)]

        # Filters
        if not self.filter_policy.allow_trading(candle):
            self.log.debug("Filters block trading at %s (close=%.6f)", candle.timestamp, candle.close)
            return []

        self._ensure_inner(candle)
        assert self._inner is not None

        return self._inner.on_candle(candle, account)

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        if self._inner is None:
            self.log.warning(
                "Received OrderFilledEvent but inner strategy is not initialised yet: %s",
                event,
            )
            return

        self._inner.on_order_filled(event)
