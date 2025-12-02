# tests/test_results_models.py
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.models import Side
from core.results.models import Trade, EquityPoint, BacktestResult


def make_dummy_result() -> BacktestResult:
    start = datetime(2025, 1, 1, 0, 0)

    equity_curve = [
        EquityPoint(timestamp=start + timedelta(minutes=i), equity=1000.0 + i * 10.0)
        for i in range(5)
    ]

    trades = [
        Trade(
            id="t1",
            symbol="XRPUSDT",
            side=Side.BUY,
            entry_time=start,
            exit_time=start + timedelta(minutes=2),
            entry_price=1.0,
            exit_price=1.1,
            size=100.0,
            gross_pnl=10.0,
            fee=0.1,
            net_pnl=9.9,
            return_pct=0.10,
            bars_held=2,
        ),
        Trade(
            id="t2",
            symbol="XRPUSDT",
            side=Side.BUY,
            entry_time=start + timedelta(minutes=2),
            exit_time=start + timedelta(minutes=4),
            entry_price=1.1,
            exit_price=1.05,
            size=100.0,
            gross_pnl=-5.0,
            fee=0.1,
            net_pnl=-5.1,
            return_pct=-0.045,
            bars_held=2,
        ),
    ]

    result = BacktestResult(
        run_id="test-run",
        run_name="Test Run",
        symbol="XRPUSDT",
        timeframe="1m",
        started_at=start,
        finished_at=start + timedelta(minutes=4),
        initial_balance=1000.0,
        final_equity=1040.0,
        trades=trades,
        equity_curve=equity_curve,
        metrics={},
        extra={},
    )
    return result


def test_backtest_result_basic_fields():
    result = make_dummy_result()

    assert result.run_id == "test-run"
    assert result.symbol == "XRPUSDT"
    assert result.initial_balance == pytest.approx(1000.0)
    assert result.final_equity == pytest.approx(1040.0)
    assert len(result.trades) == 2
    assert len(result.equity_curve) == 5

    # types
    assert all(isinstance(t, Trade) for t in result.trades)
    assert all(isinstance(p, EquityPoint) for p in result.equity_curve)


def test_backtest_result_equity_curve_is_monotonic_in_time():
    result = make_dummy_result()
    timestamps = [p.timestamp for p in result.equity_curve]
    assert timestamps == sorted(timestamps)


def test_trade_has_expected_pnl_signs():
    result = make_dummy_result()
    pnl_values = [t.net_pnl for t in result.trades]

    assert pnl_values[0] > 0     # first trade profitable
    assert pnl_values[1] < 0     # second trade losing
