# core/results/summary.py
from __future__ import annotations

from typing import Optional
import logging

from .models import BacktestResult

import pandas as pd


def print_result_summary(result: BacktestResult, logger: Optional[logging.Logger] = None) -> None:
    """
    Print a human-readable summary of the backtest.
    If logger is given, uses logger.info; otherwise, prints to stdout.
    """
    out = logger.info if logger else print

    out("")
    out("=== Backtest Summary ===")
    out(f"Run:       {result.run_name} ({result.run_id})")
    out(f"Symbol:    {result.symbol}")
    out(f"Timeframe: {result.timeframe}")
    out(f"Start:     {result.started_at.isoformat()}")
    out(f"End:       {result.finished_at.isoformat()}")
    out(f"Equity:    {result.initial_equity:.2f} -> {result.final_equity:.2f}")

    if result.metrics:
        out("")
        out("Metrics:")
        for name, value in sorted(result.metrics.items()):
            # try to format numbers nicely
            if isinstance(value, float):
                out(f"  {name:20s} {value: .4f}")
            else:
                out(f"  {name:20s} {value}")
    else:
        out("")
        out("No metrics computed.")

    out("")
    out(f"Trades:    {len(result.trades)}")
    out(f"Equity points: {len(result.equity_curve)}")
    out("")


def result_to_dataframes(result: BacktestResult):
    """
    Convert BacktestResult into two DataFrames:
      - trades_df
      - equity_df
    """
    trades_rows = [
        {
            "trade_id": t.id,
            "symbol": t.symbol,
            "side": getattr(t.side, "value", str(t.side)),
            "size": t.size,
            "entry_price": t.entry_price,
            "entry_time": t.entry_time,
            "exit_price": t.exit_price,
            "exit_time": t.exit_time,
            "gross_pnl": t.gross_pnl,
            "net_pnl": t.net_pnl,
            "fee": t.fee,
            "return_pct": t.return_pct,
            "bars_held": t.bars_held,
        }
        for t in result.trades
    ]

    equity_rows = [
        {
            "time": p.timestamp,
            "equity": p.equity,
        }
        for p in result.equity_curve
    ]

    trades_df = pd.DataFrame(trades_rows)
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df
