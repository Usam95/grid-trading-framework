# core/strategy/grid_strategy_simple.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from core.models import Candle, AccountState, Order, Side, OrderType, OrderFilledEvent
from core.strategy.base import BaseStrategy
from infra.logging_setup import get_logger


class GridSpacing(str, Enum):
    """How grid prices are spaced between lower and upper."""
    ARITHMETIC = "arithmetic"   # equal absolute distance
    GEOMETRIC = "geometric"     # equal ratio (percentage step)


class GridRangeMode(str, Enum):
    """How the price range is defined."""
    PERCENT = "percent"    # via lower_pct / upper_pct around start price
    ABSOLUTE = "absolute"  # via lower_price / upper_price directly


@dataclass
class GridConfig:
    symbol: str
    base_order_size: float
    n_levels: int  # number of price levels (grid lines)

    # --- percent-based range (around first candle close) ---
    lower_pct: Optional[float] = None   # e.g. 0.10 = 10% below start price
    upper_pct: Optional[float] = None   # e.g. 0.10 = 10% above start price

    # --- absolute price range (Binance UI: Lower / Upper) ---
    lower_price: Optional[float] = None
    upper_price: Optional[float] = None

    # how to interpret the above
    range_mode: GridRangeMode = GridRangeMode.PERCENT

    # grid spacing type (Binance: Arithmetic / Geometric)
    spacing: GridSpacing = GridSpacing.ARITHMETIC


@dataclass
class GridLevel:
    price: float
    side: Side
    size: float
    active: bool = True


class SimpleGridStrategy(BaseStrategy):
    """
    Very simple static grid for MVP backtesting.

    Behaviour:
      - On first candle:
          * Builds N grid levels within configured price range.
          * Levels below close -> BUY limit orders.
          * Levels at/above close -> SELL limit orders.
      - After that:
          * Does not create new orders when levels are filled (no re-placement yet).
    """

    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self._initialized = False
        self._grid_levels: List[GridLevel] = []
        self._order_to_level: Dict[str, GridLevel] = {}
        self.log = get_logger("strategy.grid")

    # ------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------
    def on_candle(self, candle: Candle, account: AccountState) -> List[Order]:
        if not self._initialized:
            if self.config.n_levels < 2:
                raise ValueError("GridConfig.n_levels must be >= 2")

            self.log.info(
                "Initializing SimpleGridStrategy for %s at first candle: ts=%s close=%.6f "
                "(n_levels=%d, lower_pct=%s, upper_pct=%s, lower_price=%s, upper_price=%s, spacing=%s, range_mode=%s, base_order_size=%.4f)",
                self.config.symbol,
                candle.timestamp,
                candle.close,
                self.config.n_levels,
                self.config.lower_pct,
                self.config.upper_pct,
                self.config.lower_price,
                self.config.upper_price,
                self.config.spacing.value,
                self.config.range_mode.value,
                self.config.base_order_size,
            )

            self._build_initial_grid(candle.close)
            orders = self._create_orders_for_active_levels()

            for o in orders:
                self.log.info(
                    "Initial grid order: id=%s side=%s price=%.6f qty=%.4f",
                    o.id,
                    o.side.value,
                    o.price,
                    o.qty,
                )

            self._initialized = True
            return orders

        # For MVP: no new orders after initialization
        self.log.debug(
            "No new grid orders on candle ts=%s close=%.6f (static grid).",
            candle.timestamp,
            candle.close,
        )
        return []

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        # Mark the corresponding grid level as inactive
        level = self._order_to_level.get(event.order_id)
        if level is not None:
            level.active = False
            self.log.info(
                "Grid level deactivated due to fill: order_id=%s side=%s price=%.6f qty=%.4f",
                event.order_id,
                event.side.value,
                event.price,
                event.qty,
            )
        else:
            self.log.debug(
                "Received OrderFilledEvent for unknown order_id=%s in SimpleGridStrategy.",
                event.order_id,
            )

        # Later: re-place or float grid here

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _compute_range(self, start_price: float) -> tuple[float, float]:
        """
        Compute (lower, upper) price bounds from config and starting price.

        - PERCENT mode: lower/upper are relative to start_price via lower_pct/upper_pct.
        - ABSOLUTE mode: lower/upper are taken directly from lower_price/upper_price.
        """
        if self.config.range_mode == GridRangeMode.PERCENT:
            if self.config.lower_pct is None or self.config.upper_pct is None:
                raise ValueError("Percent range_mode requires lower_pct and upper_pct.")
            lower = start_price * (1.0 - self.config.lower_pct)
            upper = start_price * (1.0 + self.config.upper_pct)

        elif self.config.range_mode == GridRangeMode.ABSOLUTE:
            if self.config.lower_price is None or self.config.upper_price is None:
                raise ValueError("Absolute range_mode requires lower_price and upper_price.")
            lower = self.config.lower_price
            upper = self.config.upper_price

        else:
            raise ValueError(f"Unknown range_mode: {self.config.range_mode}")

        if lower <= 0.0 or upper <= 0.0:
            raise ValueError("Price range must be > 0.")
        if upper <= lower:
            raise ValueError("upper must be greater than lower.")

        self.log.debug(
            "Computed grid range for %s: lower=%.6f upper=%.6f from start_price=%.6f",
            self.config.symbol,
            lower,
            upper,
            start_price,
        )

        return lower, upper

    def _build_initial_grid(self, start_price: float) -> None:
        lower, upper = self._compute_range(start_price)

        if self.config.spacing == GridSpacing.GEOMETRIC and lower <= 0.0:
            raise ValueError("Geometric spacing requires lower price > 0.")

        self._grid_levels.clear()

        # number of levels (price lines) = n_levels
        if self.config.spacing == GridSpacing.ARITHMETIC:
            # Equal absolute step between prices, including lower and upper
            step = (upper - lower) / (self.config.n_levels - 1)
            for i in range(self.config.n_levels):
                price = lower + i * step
                side = Side.BUY if price < start_price else Side.SELL
                level = GridLevel(
                    price=price,
                    side=side,
                    size=self.config.base_order_size,
                )
                self._grid_levels.append(level)

        elif self.config.spacing == GridSpacing.GEOMETRIC:
            # Equal ratio between successive prices, including lower and upper
            ratio = (upper / lower) ** (1.0 / (self.config.n_levels - 1))
            for i in range(self.config.n_levels):
                price = lower * (ratio ** i)
                side = Side.BUY if price < start_price else Side.SELL
                level = GridLevel(
                    price=price,
                    side=side,
                    size=self.config.base_order_size,
                )
                self._grid_levels.append(level)

        else:
            raise ValueError(f"Unknown spacing type: {self.config.spacing}")

        self.log.info(
            "Built initial grid for %s with %d levels in [%0.6f, %0.6f].",
            self.config.symbol,
            len(self._grid_levels),
            lower,
            upper,
        )

    def _create_orders_for_active_levels(self) -> List[Order]:
        orders: List[Order] = []
        for level in self._grid_levels:
            if not level.active:
                continue
            order_id = str(uuid4())
            order = Order(
                id=order_id,
                symbol=self.config.symbol,
                side=level.side,
                price=level.price,
                qty=level.size,
                type=OrderType.LIMIT,
            )
            self._order_to_level[order_id] = level
            orders.append(order)

        self.log.info(
            "Created %d orders for active grid levels (%s).",
            len(orders),
            self.config.symbol,
        )
        return orders
