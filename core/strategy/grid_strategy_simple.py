from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from core.engine_actions import EngineAction
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
    n_levels: int

    lower_pct: Optional[float] = None
    upper_pct: Optional[float] = None
    range_mode: GridRangeMode = GridRangeMode.PERCENT

    lower_price: Optional[float] = None
    upper_price: Optional[float] = None

    spacing: GridSpacing = GridSpacing.ARITHMETIC


@dataclass
class GridLevel:
    price: float
    side: Side
    size: float
    active: bool = True


class SimpleGridStrategy(BaseStrategy):
    """
    Binance-like static spot grid strategy.

    IMPORTANT FIX:
      - We no longer store Order objects as dict keys (Order is mutable/unhashable).
      - Instead, we use Order.client_tag -> GridLevel mapping.
      - Engine propagates client_tag into OrderFilledEvent, so we can map fills.
    """

    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self.log = get_logger("core.strategy.grid.simple")

        self._initialized = False
        self._grid_levels: List[GridLevel] = []

        # FIX: tag -> level mapping (hashable stable key)
        self._tag_to_level: Dict[str, GridLevel] = {}

        self._lower: Optional[float] = None
        self._upper: Optional[float] = None

        self._pending_orders: List[Order] = []

    # ------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------
    def on_candle(self, candle: Candle, account: AccountState) -> List[EngineAction]:
        if not self._initialized:
            self.log.info(
                "Initializing Binance-style GridStrategy for %s at first candle: "
                "ts=%s close=%.6f (n_levels=%d)",
                self.config.symbol,
                candle.timestamp,
                candle.close,
                self.config.n_levels,
            )

            self._build_initial_grid(candle.close)
            orders = self._seed_initial_orders(candle.close)

            for o in orders:
                self.log.info(
                    "Initial grid order: id=%s tag=%s side=%s price=%.6f qty=%.4f",
                    o.id,  # filled by engine later
                    o.client_tag,
                    o.side.value,
                    o.price,
                    o.qty,
                )

            self._initialized = True
            return [EngineAction.place(o) for o in orders]

        if self._pending_orders:
            orders = self._pending_orders
            self._pending_orders = []
            self.log.debug(
                "Emitting %d pending neighbor-orders at ts=%s close=%.6f",
                len(orders),
                candle.timestamp,
                candle.close,
            )
            return [EngineAction.place(o) for o in orders]

        return []

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        tag = event.client_tag
        if not tag:
            self.log.debug(
                "Received OrderFilledEvent without client_tag (order_id=%s). Cannot map to grid level.",
                event.order_id,
            )
            return

        level = self._tag_to_level.get(tag)
        if level is None:
            self.log.debug(
                "Received OrderFilledEvent for unknown client_tag=%s (order_id=%s).",
                tag,
                event.order_id,
            )
            return

        level.active = False

        self.log.debug(
            "Grid level filled: order_id=%s tag=%s side=%s price=%.6f qty=%.4f",
            event.order_id,
            tag,
            event.side.value,
            event.price,
            event.qty,
        )

        if event.side == Side.BUY:
            self._place_sell_neighbor(level)
        elif event.side == Side.SELL:
            self._place_buy_neighbor(level)

    # ------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------
    def _compute_range(self, start_price: float) -> tuple[float, float]:
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

        return lower, upper

    def _build_initial_grid(self, start_price: float) -> None:
        lower, upper = self._compute_range(start_price)
        self._lower, self._upper = lower, upper

        n = self.config.n_levels
        if self.config.spacing == GridSpacing.ARITHMETIC:
            step = (upper - lower) / (n - 1)
            prices = [lower + i * step for i in range(n)]
        else:
            ratio = (upper / lower) ** (1.0 / (n - 1))
            prices = [lower * (ratio ** i) for i in range(n)]

        self._grid_levels = [
            GridLevel(price=p, side=Side.BUY, size=self.config.base_order_size, active=False)
            for p in prices
        ]

    def _level_index(self, level: GridLevel) -> int:
        return self._grid_levels.index(level)

    def _level_tag(self, level: GridLevel) -> str:
        idx = self._level_index(level)
        return f"{self.config.symbol}:LVL:{idx}"

    def _create_order_for_level(self, level: GridLevel) -> Order:
        tag = self._level_tag(level)

        order = Order(
            id="",  # ENGINE assigns ID as SYMBOL-N (e.g. XRPUSDT-1)
            symbol=self.config.symbol,
            side=level.side,
            price=level.price,
            qty=level.size,
            type=OrderType.LIMIT,
            client_tag=tag,
        )

        # Update mapping so we can map fills -> level
        self._tag_to_level[tag] = level
        return order

    def _seed_initial_orders(self, start_price: float) -> List[Order]:
        import bisect
        prices = [lvl.price for lvl in self._grid_levels]
        # k = index of last grid price strictly below start_price
        k = bisect.bisect_left(prices, start_price) - 1

        orders: List[Order] = []

        for i in range(0, k + 1):
            lvl = self._grid_levels[i]
            lvl.side = Side.BUY
            lvl.active = True
            orders.append(self._create_order_for_level(lvl))

        for i in range(k + 2, len(self._grid_levels)):
            lvl = self._grid_levels[i]
            lvl.side = Side.SELL
            lvl.active = True
            orders.append(self._create_order_for_level(lvl))

        return orders

    def _place_sell_neighbor(self, level: GridLevel) -> None:
        idx = self._level_index(level)
        if idx + 1 >= len(self._grid_levels):
            return

        neighbor = self._grid_levels[idx + 1]
        if neighbor.active:
            return

        neighbor.side = Side.SELL
        neighbor.active = True
        self._pending_orders.append(self._create_order_for_level(neighbor))

    def _place_buy_neighbor(self, level: GridLevel) -> None:
        idx = self._level_index(level)
        if idx - 1 < 0:
            return

        neighbor = self._grid_levels[idx - 1]
        if neighbor.active:
            return

        neighbor.side = Side.BUY
        neighbor.active = True
        self._pending_orders.append(self._create_order_for_level(neighbor))
