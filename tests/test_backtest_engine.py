# tests/test_backtest_engine.py
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from core.models import Candle, Side, OrderType, Order, AccountState
from core.strategy.base import IStrategy

from backtest.engine import BacktestEngine
from infra.config_models import BacktestEngineConfig, LocalDataConfig


# ---------------------------------------------------------------------------
# Helpers / Test Doubles
# ---------------------------------------------------------------------------


class FakeDataSource:
    """
    Minimal stub that mimics LocalFileDataSource.load(cfg).

    It ignores the config and always returns the injected DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.last_cfg = None  # for assertions if needed

    def load(self, cfg) -> pd.DataFrame:
        # cfg in real code is DatasetConfig, coming from LocalDataConfig
        self.last_cfg = cfg
        return self._df


class OneShotBuyStrategy(IStrategy):
    """
    Simple test strategy:

    - On first candle: place one LIMIT BUY at close price (qty = 1).
    - Afterwards: no more orders.
    - Records fills passed to on_order_filled.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._first_call_done = False
        self.fills = []

    def on_candle(self, candle: Candle, account: AccountState):
        if self._first_call_done:
            return []

        self._first_call_done = True

        order = Order(
            id="test-order-1",
            symbol=self.symbol,
            side=Side.BUY,
            price=candle.close,  # equal to close so it's fillable in our simple fill model
            qty=1.0,
            type=OrderType.LIMIT,
        )
        return [order]

    def on_order_filled(self, event, *_, **__):
        # The engine must call this when orders are filled
        self.fills.append(event)


def make_test_dataframe(n: int = 5, start_price: float = 10.0) -> pd.DataFrame:
    """
    Creates a simple 1-minute OHLCV DataFrame.
    """
    idx = pd.date_range("2025-01-01 00:00:00", periods=n, freq="1min")
    prices = [start_price + i * 0.1 for i in range(n)]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 0.05 for p in prices],
            "Low": [p - 0.05 for p in prices],
            "Close": prices,
            "Volume": [1000.0] * n,
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_engine_with_df(df: pd.DataFrame) -> tuple[BacktestEngine, OneShotBuyStrategy]:
    """
    Small helper to build a BacktestEngine wired to FakeDataSource and OneShotBuyStrategy.
    """
    ds = FakeDataSource(df)
    engine_cfg = BacktestEngineConfig(
        initial_balance=1000.0,
        trading_fee_pct=0.0,
        slippage_pct=0.0,
    )
    data_cfg = LocalDataConfig(
        symbol="XRPUSDT",
        start=None,
        end=None,
    )
    strat = OneShotBuyStrategy(symbol="XRPUSDT")

    engine = BacktestEngine(
        engine_cfg=engine_cfg,
        data_cfg=data_cfg,
        strategy=strat,
        data_source=ds,
    )
    return engine, strat


def test_to_candles_converts_dataframe_correctly():
    """
    BacktestEngine._to_candles should map OHLCV DataFrame → list[Candle].
    """
    df = make_test_dataframe(n=3, start_price=10.0)
    engine, _ = _make_engine_with_df(df)

    # private helper but fine for tests
    candles = engine._to_candles(df)  # type: ignore[attr-defined]

    assert len(candles) == len(df)
    assert isinstance(candles[0], Candle)
    # Timestamp mapping
    assert candles[0].timestamp == df.index[0].to_pydatetime()
    # OHLCV mapping
    assert candles[0].open == df["Open"].iloc[0]
    assert candles[0].high == df["High"].iloc[0]
    assert candles[0].low == df["Low"].iloc[0]
    assert candles[0].close == df["Close"].iloc[0]
    assert candles[0].volume == df["Volume"].iloc[0]


def test_engine_runs_and_executes_oneshot_buy_strategy():
    """
    Smoke test:

    - Engine runs over a small DataFrame
    - Strategy is called
    - One LIMIT BUY is placed and filled
    - Account cash balance is reduced by notional
    - Strategy.on_order_filled is called.
    """
    df = make_test_dataframe(n=5, start_price=10.0)
    engine, strat = _make_engine_with_df(df)

    result = engine.run()

    # Strategy should have recorded exactly one fill
    assert len(strat.fills) == 1
    fill = strat.fills[0]
    assert fill.symbol == "XRPUSDT"
    assert fill.side == Side.BUY
    # Filled at candle.close of the first bar (10.0)
    assert fill.price == pytest.approx(10.0)

    # After one BUY of qty=1 at 10.0 and zero fees:
    #  - invested_cost should be 10.0
    #  - cash should be 990.0
    assert engine.account.cash_balance == pytest.approx(1000.0 - 10.0)
    assert engine.account.invested_cost == pytest.approx(10.0)

    # Equity should be cash + market value of open position at last candle close
    last_close = df["Close"].iloc[-1]
    expected_market_value = last_close * 1.0
    expected_total = engine.account.cash_balance + expected_market_value
    assert engine.account.total_value == pytest.approx(expected_total)

    # BacktestResult basic sanity checks
    assert result.initial_equity == pytest.approx(1000.0)
    assert result.final_equity == pytest.approx(engine.account.total_value)
    # We should have at least one equity point
    assert len(result.equity_curve) > 0
