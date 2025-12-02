# backtest/results_models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

from core.models import Side


@dataclass
class EquityPoint:
    """
    Single point on the equity curve.
    """
    # tests use: EquityPoint(timestamp=..., equity=...)
    timestamp: datetime
    equity: float


@dataclass
class Trade:
    """
    Summary of a completed trade.
    """
    # tests use: Trade(id="t1", ...)
    id: str
    symbol: str
    side: Side

    entry_time: datetime
    exit_time: datetime

    entry_price: float
    exit_price: float
    size: float

    gross_pnl: float
    fee: float
    net_pnl: float
    return_pct: float
    bars_held: int


@dataclass
class BacktestResult:
    """
    High-level summary of a single backtest run.

    Field names are aligned with tests/test_results_models.py.
    """
    run_id: str
    run_name: str
    symbol: str
    timeframe: str

    started_at: datetime
    finished_at: datetime

    # IMPORTANT: tests pass 'initial_balance', not 'starting_equity' or 'initial_equity'
    initial_balance: float
    final_equity: float

    trades: List[Trade]
    equity_curve: List[EquityPoint]

    # Optional / extra info – tests also pass these, but content is not constrained
    metrics: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # --- Backwards compatible alias for tests/old code ---
    @property
    def initial_equity(self) -> float:
        """Alias for initial_balance (for older code/tests)."""
        return self.initial_balance

    @initial_equity.setter
    def initial_equity(self, value: float) -> None:
        self.initial_balance = value