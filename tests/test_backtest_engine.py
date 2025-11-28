# tests/test_backtest_engine.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pytest

from core.models import (
    Candle,
    Side,
    OrderType,
    Order,
    AccountState,
)
from backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
)
from core.strategy.base import IStrategy


# ---------------------------------------------------------------------------
# Helpers / Test Doubles
# ---------------------------------------------------------------------------

class FakeDataSource:
    """
    Minimal stub that mimics LocalFileDataSource.load().
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def load(self, cfg) -> pd.DataFrame:
        # ignore cfg.start/cfg.end for simplicity — df is already in desired range
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
        self.fills: list = []

    def on_candle(self, candle: Candle, account: AccountState):
        if self._first_call_done:
            return []

        self._first_call_done = True

        order = Order(
            id="test-order-1",
            symbol=self.symbol,
            side=Side.BUY,
            price=candle.close,  # equal to close so it's fillable in simple fill model
            qty=1.0,
            type=OrderType.LIMIT,
        )
        return [order]

    def on_order_filled(self, event):
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

def test_to_candles_converts_dataframe_correctly():
    df = make_test_dataframe(n=3, start_price=10.0)
    ds = FakeDataSource(df)
    cfg = BacktestConfig(
        symbol="XRPUSDT",
        start=df.index[0].isoformat(),
        end=df.index[-1].isoformat(),
        initial_balance=1000.0,
        trading_fee_pct=0.0,
    )
    strat = OneShotBuyStrategy(symbol="XRPUSDT")
    engine = BacktestEngine(config=cfg, data_source=ds, strategy=strat)

    candles = engine._to_candles(df)

    assert len(candles) == len(df)
    assert isinstance(candles[0], Candle)
    assert candles[0].timestamp == df.index[0]
    assert candles[0].open == df["Open"].iloc[0]
    assert candles[0].close == df["Close"].iloc[0]
    assert candles[0].volume == df["Volume"].iloc[0]


def test_engine_run_produces_backtest_result_and_equity_curve():
    df = make_test_dataframe(n=10, start_price=10.0)
    ds = FakeDataSource(df)
    cfg = BacktestConfig(
        symbol="XRPUSDT",
        start=df.index[0].isoformat(),
        end=df.index[-1].isoformat(),
        initial_balance=1000.0,
        trading_fee_pct=0.0,
    )
    strat = OneShotBuyStrategy(symbol="XRPUSDT")
    engine = BacktestEngine(config=cfg, data_source=ds, strategy=strat)

    result = engine.run()

    # --- basic structure checks ---
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == len(df)

    # At least one equity point should differ from initial balance
    initial_balance = cfg.initial_balance
    equity_values = [eq for (_, eq) in result.equity_curve]
    assert any(abs(eq - initial_balance) > 1e-9 for eq in equity_values)

    # Strategy should have received at least one fill
    assert len(strat.fills) >= 1


def test_engine_respects_initial_balance():
    """
    Engine should start from initial_balance, then apply trades.

    For OneShotBuyStrategy:
      - first candle: buy 1 unit at first close
      - equity after first candle = initial_balance - close[0]
    """
    df = make_test_dataframe(n=3, start_price=10.0)
    ds = FakeDataSource(df)
    initial_balance = 500.0
    cfg = BacktestConfig(
        symbol="XRPUSDT",
        start=df.index[0].isoformat(),
        end=df.index[-1].isoformat(),
        initial_balance=initial_balance,
        trading_fee_pct=0.0,
    )
    strat = OneShotBuyStrategy(symbol="XRPUSDT")
    engine = BacktestEngine(config=cfg, data_source=ds, strategy=strat)

    result = engine.run()

    first_ts, first_equity = result.equity_curve[0]

    first_close = df["Close"].iloc[0]
    expected_first_equity = initial_balance - first_close  # 500 - 10 = 490 in current setup

    assert pytest.approx(first_equity, rel=1e-9) == expected_first_equity
