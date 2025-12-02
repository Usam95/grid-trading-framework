# tests/test_grid_strategy_simple.py
from __future__ import annotations

from datetime import datetime
import inspect

from core.models import (
    Candle,
    AccountState,
    Side,
    OrderFilledEvent,
)
from core.strategy.grid_strategy_simple import (
    GridConfig,
    SimpleGridStrategy,
)


def make_candle(price: float = 1.0) -> Candle:
    return Candle(
        timestamp=datetime(2025, 1, 1, 0, 0),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000.0,
    )


def make_account(balance: float = 1_000.0) -> AccountState:
    # Use positional args to avoid coupling to internal field names.
    return AccountState(balance, balance, 0.0)


def test_initial_grid_builds_orders_on_first_candle():
    cfg = GridConfig(
        symbol="XRPUSDT",
        base_order_size=10.0,
        n_levels=3,
        lower_pct=0.03,  # 3% below
        upper_pct=0.03,  # 3% above
    )
    strat = SimpleGridStrategy(config=cfg)

    candle = make_candle(price=1.0)
    account = make_account()

    orders = strat.on_candle(candle, account)

    # --- basic sanity checks ---
    assert len(orders) == cfg.n_levels

    # All orders for same symbol and quantity
    assert all(o.symbol == cfg.symbol for o in orders)
    assert all(o.qty == cfg.base_order_size for o in orders)

    # We expect both BUY and SELL orders present
    sides = {o.side for o in orders}
    assert Side.BUY in sides
    assert Side.SELL in sides

    # Prices should span below and above current price
    prices = [o.price for o in orders]
    assert min(prices) < candle.close
    assert max(prices) > candle.close


def test_second_candle_does_not_rebuild_grid():
    cfg = GridConfig(
        symbol="XRPUSDT",
        base_order_size=10.0,
        n_levels=2,
        lower_pct=0.02,
        upper_pct=0.02,
    )
    strat = SimpleGridStrategy(config=cfg)

    candle1 = make_candle(price=1.0)
    candle2 = make_candle(price=1.01)
    account = make_account()

    orders_first = strat.on_candle(candle1, account)
    assert len(orders_first) > 0  # grid created

    orders_second = strat.on_candle(candle2, account)
    # For static grid MVP we expect no new orders on subsequent candles
    assert orders_second == []


def test_on_order_filled_deactivates_grid_level():
    cfg = GridConfig(
        symbol="XRPUSDT",
        base_order_size=5.0,
        n_levels=2,   # must be >= 2 for your implementation
        lower_pct=0.01,
        upper_pct=0.01,
    )
    strat = SimpleGridStrategy(config=cfg)

    candle = make_candle(price=1.0)
    account = make_account()

    # First call: build grid and get initial orders
    orders = strat.on_candle(candle, account)
    assert len(orders) == cfg.n_levels

    # Pick one order to simulate a fill
    filled_order = orders[0]

    # Ensure this order is tracked in internal mapping
    assert filled_order.id in strat._order_to_level

    grid_level_before = strat._order_to_level[filled_order.id]
    assert grid_level_before.active is True

    event = OrderFilledEvent(
        order_id=filled_order.id,
        symbol=filled_order.symbol,
        side=filled_order.side,
        price=filled_order.price,
        qty=filled_order.qty,
        filled_at=datetime(2025, 1, 1, 0, 1),
        position_id=None,
    )

    # Support both (event) and (event, account) signatures
    sig = inspect.signature(strat.on_order_filled)
    if len(sig.parameters) == 1:
        strat.on_order_filled(event)
    else:
        strat.on_order_filled(event, account)

    grid_level_after = strat._order_to_level[filled_order.id]
    assert grid_level_after.active is False
