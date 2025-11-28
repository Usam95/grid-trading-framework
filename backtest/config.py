# backtest/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """
    Configuration for a single-symbol backtest run.
    """
    symbol: str                # e.g. "XRPUSDT"
    start: str                 # e.g. "2025-11-01 00:00:00"
    end: str                   # e.g. "2025-11-20 23:59:00"
    initial_balance: float     # e.g. 10_000.0 (USDT)
    trading_fee_pct: float = 0.00075  # per-side fee, e.g. 0.075%
