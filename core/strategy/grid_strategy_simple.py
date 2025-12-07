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
    Binance-style neighbor grid (fixed band, no trailing / floating yet).

    Behaviour:
      - On first candle:
          * Compute price range [lower, upper] from config.
          * Build N grid price levels between lower and upper.
          * Seed initial orders like Binance:
              - Let prices = p0 < p1 < ... < pn.
              - Find k = max i where p[i] < start_price.
              - Place BUY at p[0..k].
              - Skip p[k+1].
              - Place SELL at p[k+2..n].
      - On fill:
          * BUY filled at p[i]  -> place SELL at p[i+1] (if exists and inactive).
          * SELL filled at p[i] -> place BUY  at p[i-1] (if exists and inactive).
      - Orders created after fills are emitted on the next candle via _pending_orders.
    """

    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self._initialized = False
        self._grid_levels: List[GridLevel] = []
        self._order_to_level: Dict[str, GridLevel] = {}
        self.log = get_logger("strategy.grid")

        # Grid spacing / band info
        self._step: Optional[float] = None   # for ARITHMETIC
        self._ratio: Optional[float] = None  # for GEOMETRIC
        self._lower: Optional[float] = None
        self._upper: Optional[float] = None

        # Orders created in on_order_filled and emitted on next candle
        self._pending_orders: List[Order] = []

    # ------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------
    def on_candle(self, candle: Candle, account: AccountState) -> List[Order]:
        if not self._initialized:
            if self.config.n_levels < 2:
                raise ValueError("GridConfig.n_levels must be >= 2")

            self.log.info(
                "Initializing Binance-style GridStrategy for %s at first candle: "
                "ts=%s close=%.6f (n_levels=%d, lower_pct=%s, upper_pct=%s, "
                "lower_price=%s, upper_price=%s, spacing=%s, range_mode=%s, base_order_size=%.4f)",
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
            orders = self._seed_initial_orders(candle.close)

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

        # After initialization: emit any pending neighbor orders
        if self._pending_orders:
            orders = self._pending_orders
            self._pending_orders = []
            self.log.debug(
                "Emitting %d pending neighbor-orders at ts=%s close=%.6f",
                len(orders),
                candle.timestamp,
                candle.close,
            )
            return orders

        self.log.debug(
            "No new grid orders on candle ts=%s close=%.6f.",
            candle.timestamp,
            candle.close,
        )
        return []

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        # Find the corresponding grid level
        level = self._order_to_level.get(event.order_id)
        if level is None:
            self.log.debug(
                "Received OrderFilledEvent for unknown order_id=%s in SimpleGridStrategy.",
                event.order_id,
            )
            return

        # Mark level as inactive (this particular order is done)
        level.active = False
        self.log.debug(
            "Grid level filled: order_id=%s side=%s price=%.6f qty=%.4f",
            event.order_id,
            event.side.value,
            event.price,
            event.qty,
        )

        # Apply Binance neighbor rule:
        #   BUY filled at p[i]  -> place SELL at p[i+1]
        #   SELL filled at p[i] -> place BUY  at p[i-1]
        if event.side == Side.BUY:
            self._on_buy_filled(level)
        elif event.side == Side.SELL:
            self._on_sell_filled(level)

    # ------------------------------------------------------------
    # Internal helpers: grid building & seeding
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
        """
        Build the list of grid price levels p0 < p1 < ... < pN.

        At this stage we only define prices; side/active flags will be
        set in _seed_initial_orders based on Binance-like rules.
        """
        lower, upper = self._compute_range(start_price)

        if self.config.spacing == GridSpacing.GEOMETRIC and lower <= 0.0:
            raise ValueError("Geometric spacing requires lower price > 0.")

        self._grid_levels.clear()

        if self.config.spacing == GridSpacing.ARITHMETIC:
            # Equal absolute step between prices, including lower and upper
            step = (upper - lower) / (self.config.n_levels - 1)
            self._step = step
            self._ratio = None

            for i in range(self.config.n_levels):
                price = lower + i * step
                level = GridLevel(
                    price=price,
                    side=Side.BUY,                 # placeholder, will be set later
                    size=self.config.base_order_size,
                    active=False,                  # inactive until seeded
                )
                self._grid_levels.append(level)

        elif self.config.spacing == GridSpacing.GEOMETRIC:
            # Equal ratio between successive prices, including lower and upper
            ratio = (upper / lower) ** (1.0 / (self.config.n_levels - 1))
            self._ratio = ratio
            self._step = None

            for i in range(self.config.n_levels):
                price = lower * (ratio ** i)
                level = GridLevel(
                    price=price,
                    side=Side.BUY,                 # placeholder
                    size=self.config.base_order_size,
                    active=False,
                )
                self._grid_levels.append(level)

        else:
            raise ValueError(f"Unknown spacing type: {self.config.spacing}")

        self._lower = lower
        self._upper = upper

        self.log.info(
            "Built initial grid for %s with %d levels in [%0.6f, %0.6f].",
            self.config.symbol,
            len(self._grid_levels),
            lower,
            upper,
        )

    def _create_order_for_level(self, level: GridLevel) -> Order:
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
        return order

    def _seed_initial_orders(self, start_price: float) -> List[Order]:
        """
        Binance-like seeding:

          Let prices = sorted grid prices p0 < p1 < ... < pN.

          - Find k = max i where p[i] < start_price.
          - Place BUY at p[0..k].
          - Skip p[k+1].
          - Place SELL at p[k+2..N].

        This approximates the behavior described in Binance docs where
        buys are below current price, sells are above, and a middle level
        around current price may start empty.
        """
        prices = [lvl.price for lvl in self._grid_levels]
        n = len(prices) - 1

        # find k: highest index where p[i] < start_price
        k: Optional[int] = None
        for i, p in enumerate(prices):
            if p < start_price:
                k = i
            else:
                break
        if k is None:
            # start price <= lowest grid -> treat as all sells above
            k = -1

        orders: List[Order] = []

        # BUYs below current (indices 0..k)
        for i in range(0, k + 1):
            lvl = self._grid_levels[i]
            lvl.side = Side.BUY
            lvl.active = True
            order = self._create_order_for_level(lvl)
            orders.append(order)

        # SELLs above current, skipping immediate neighbor (k+1)
        for i in range(k + 2, n + 1):
            lvl = self._grid_levels[i]
            lvl.side = Side.SELL
            lvl.active = True
            order = self._create_order_for_level(lvl)
            orders.append(order)

        self.log.info(
            "Seeded initial Binance-style grid: %d BUY levels (0..%d), %d SELL levels (%d..%d), skipped index %d.",
            k + 1,
            k,
            max(0, n - (k + 1)),
            k + 2 if k + 2 <= n else -1,
            n,
            k + 1,
        )
        return orders

    # ------------------------------------------------------------
    # Internal helpers: neighbor rule
    # ------------------------------------------------------------
    def _level_index(self, level: GridLevel) -> int:
        """Return the index of the level in _grid_levels."""
        for i, lvl in enumerate(self._grid_levels):
            if lvl is level:
                return i
        raise ValueError("Level not found in _grid_levels.")

    def _on_buy_filled(self, level: GridLevel) -> None:
        """
        BUY filled at p[i] -> place SELL at p[i+1] (if exists and inactive).
        """
        idx = self._level_index(level)
        next_idx = idx + 1
        if next_idx >= len(self._grid_levels):
            # no grid above -> at upper edge
            self.log.debug(
                "BUY filled at topmost level idx=%d price=%.6f, no neighbor above.",
                idx,
                level.price,
            )
            return

        neighbor = self._grid_levels[next_idx]
        if neighbor.active:
            # already has an order, don't double-book
            self.log.debug(
                "BUY filled at idx=%d price=%.6f, neighbor SELL at idx=%d price=%.6f already active.",
                idx,
                level.price,
                next_idx,
                neighbor.price,
            )
            return

        neighbor.side = Side.SELL
        neighbor.active = True
        order = self._create_order_for_level(neighbor)
        self._pending_orders.append(order)

        self.log.debug(
            "Neighbor rule: BUY filled at %.6f (idx=%d) -> placed SELL at neighbor %.6f (idx=%d)",
            level.price,
            idx,
            neighbor.price,
            next_idx,
        )

    def _on_sell_filled(self, level: GridLevel) -> None:
        """
        SELL filled at p[i] -> place BUY at p[i-1] (if exists and inactive).
        """
        idx = self._level_index(level)
        prev_idx = idx - 1
        if prev_idx < 0:
            # no grid below -> lower edge
            self.log.debug(
                "SELL filled at lowest level idx=%d price=%.6f, no neighbor below.",
                idx,
                level.price,
            )
            return

        neighbor = self._grid_levels[prev_idx]
        if neighbor.active:
            self.log.debug(
                "SELL filled at idx=%d price=%.6f, neighbor BUY at idx=%d price=%.6f already active.",
                idx,
                level.price,
                prev_idx,
                neighbor.price,
            )
            return

        neighbor.side = Side.BUY
        neighbor.active = True
        order = self._create_order_for_level(neighbor)
        self._pending_orders.append(order)

        self.log.debug(
            "Neighbor rule: SELL filled at %.6f (idx=%d) -> placed BUY at neighbor %.6f (idx=%d)",
            level.price,
            idx,
            neighbor.price,
            prev_idx,
        )
