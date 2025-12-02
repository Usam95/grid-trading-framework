# tests/test_data_source_and_models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, List

import pandas as pd

from core.models import (
    Candle,
    Order,
    Position,
    AccountState,
    OrderFilledEvent,
    Side,
    OrderType,
    PositionSide,
)
from infra.data_source import LocalFileDataSource
from infra.config_models import LocalDataConfig


# -----------------------
# Helpers
# -----------------------

def df_to_candles(df: pd.DataFrame, limit: Optional[int] = None) -> List[Candle]:
    """
    Convert a DataFrame with OHLCV columns into a list of Candle objects.
    Used only in tests for now.
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    assert not missing, f"DataFrame is missing required columns: {missing}"

    if limit is not None:
        df = df.iloc[:limit]

    candles: List[Candle] = []
    for ts, row in df.iterrows():
        if not isinstance(ts, datetime):
            ts = pd.to_datetime(ts)

        candles.append(
            Candle(
                timestamp=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
        )
    return candles


# -----------------------
# core/models.py tests
# -----------------------

def test_candle_basic():
    ts = datetime(2025, 1, 1, 12, 0)
    c = Candle(timestamp=ts, open=1.0, high=2.0, low=0.5, close=1.5, volume=1000.0)

    assert c.timestamp == ts
    assert c.open == 1.0
    assert c.high == 2.0
    assert c.low == 0.5
    assert c.close == 1.5
    assert c.volume == 1000.0


def test_order_basic():
    o = Order(
        id="ord-1",
        symbol="XRPUSDT",
        side=Side.BUY,
        price=0.55,
        qty=100.0,
        type=OrderType.LIMIT,
    )

    assert o.id == "ord-1"
    assert o.symbol == "XRPUSDT"
    assert o.side is Side.BUY
    assert o.type is OrderType.LIMIT
    assert o.qty == 100.0
    assert o.price == 0.55
    assert o.is_active is True


def test_position_open_and_close():
    opened = datetime(2025, 1, 1, 12, 0)
    p = Position(
        id="pos-1",
        symbol="XRPUSDT",
        side=PositionSide.LONG,
        entry_price=0.5,
        size=100.0,
        opened_at=opened,
    )

    assert p.is_open
    assert p.closed_at is None

    p.closed_at = datetime(2025, 1, 2, 12, 0)
    assert not p.is_open


def test_account_state_can_be_created():
    """
    AccountState fields may change over time.
    We just ensure it can be instantiated with three numeric args.
    """
    acc = AccountState(1000.0, 1000.0, 0.0)
    assert isinstance(acc, AccountState)


def test_order_filled_event():
    ts = datetime(2025, 1, 1, 12, 0)
    evt = OrderFilledEvent(
        order_id="ord-1",
        symbol="XRPUSDT",
        side=Side.BUY,
        price=0.5,
        qty=50.0,
        filled_at=ts,
        position_id="pos-1",
    )

    assert evt.order_id == "ord-1"
    assert evt.symbol == "XRPUSDT"
    assert evt.side is Side.BUY
    assert evt.qty == 50.0
    assert evt.position_id == "pos-1"


# -----------------------
# Data loading + mapping
# -----------------------

def test_load_xrpusdt_full_range():
    """
    Integration test: load XRPUSDT from local storage.
    Assumes you have historical_data/XRPUSDT/XRPUSDT.parquet.gzip.
    """
    ds = LocalFileDataSource()
    cfg = LocalDataConfig(symbol="XRPUSDT", start=None, end=None)

    df = ds.load(cfg)

    assert not df.empty, "Loaded DataFrame for XRPUSDT is empty."
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns, f"Column {col} missing in XRPUSDT data."


def test_load_xrpusdt_partial_range():
    """
    Test that start/end clipping works.
    Adjust dates if your dataset starts later.
    """
    ds = LocalFileDataSource()
    cfg = LocalDataConfig(
        symbol="XRPUSDT",
        start="2025-11-01T00:00:00",
        end="2025-11-05T23:59:00",
    )

    df = ds.load(cfg)

    assert not df.empty
    assert df.index[0] >= pd.to_datetime("2025-11-01 00:00:00")
    assert df.index[-1] <= pd.to_datetime("2025-11-05 23:59:00")


def test_convert_xrpusdt_to_candles():
    """
    Convert the first N rows to Candle objects.
    """
    ds = LocalFileDataSource()
    cfg = LocalDataConfig(symbol="XRPUSDT", start=None, end=None)
    df = ds.load(cfg)

    candles = df_to_candles(df, limit=10)
    assert len(candles) == 10

    first = candles[0]
    assert isinstance(first.timestamp, datetime)
    assert first.open > 0
    assert first.high > 0
    assert first.low > 0
    assert first.close > 0
    assert first.volume >= 0


def test_xrpusdt_looks_like_1m_data():
    """
    If XRPUSDT was downloaded with interval='1m',
    median step should be ~60 seconds.
    """
    ds = LocalFileDataSource()
    cfg = LocalDataConfig(symbol="XRPUSDT", start=None, end=None)
    df = ds.load(cfg)

    diffs = df.index.to_series().diff().dropna()
    median_delta = diffs.median()
    # ~60 seconds with small tolerance
    assert abs(median_delta.total_seconds() - 60) < 1.0
