# core/results/repository.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from .models import BacktestResult, Trade, EquityPoint


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _serialize_enum(value: Any) -> Any:
    # For enums like Side.BUY -> "BUY"
    if hasattr(value, "value"):
        return value.value
    return str(value)


def save_backtest_result(result: BacktestResult, base_dir: Path | str = "results") -> Path:
    """
    Save a BacktestResult into:
      results/<run_id>/
        - summary.json
        - trades.csv
        - equity_curve.csv
        - extra.json (optional)
    """
    base_dir = Path(base_dir)
    run_dir = base_dir / result.run_id
    _ensure_dir(run_dir)

    # ------------------------------------------------------------------
    # 1) Summary (JSON)
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "run_id": result.run_id,
        "run_name": result.run_name,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "initial_equity": result.initial_equity,
        "final_equity": result.final_equity,
        "metrics": result.metrics,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # 2) Trades (CSV)
    # ------------------------------------------------------------------
    trade_fields = [
        "trade_id",
        "symbol",
        "side",
        "size",
        "entry_price",
        "entry_time",
        "exit_price",
        "exit_time",
        "realized_pnl",
        "fee",
    ]

    with (run_dir / "trades.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trade_fields)
        writer.writeheader()
        for t in result.trades:
            # assuming Trade(dataclass) with fields as defined in models.py
            writer.writerow(
                {
                    "trade_id": t.id,
                    "symbol": t.symbol,
                    "side": _serialize_enum(t.side),
                    "size": t.size,
                    "entry_price": t.entry_price,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_price": t.exit_price,
                    "exit_time": t.exit_time.isoformat(),
                    "realized_pnl": t.net_pnl,
                    "fee": getattr(t, "fee", 0.0),
                }
            )

    # ------------------------------------------------------------------
    # 3) Equity curve (CSV)
    # ------------------------------------------------------------------
    with (run_dir / "equity_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "equity"])
        for p in result.equity_curve:
            writer.writerow([p.timestamp.isoformat(), p.equity])

    # ------------------------------------------------------------------
    # 4) Extra data (JSON, if present)
    # ------------------------------------------------------------------
    if result.extra:
        def default(o: Any) -> Any:
            # best-effort serializer for misc objects
            if is_dataclass(o):
                return asdict(o)
            if hasattr(o, "isoformat"):
                return o.isoformat()
            if hasattr(o, "value"):
                return o.value
            return str(o)

        with (run_dir / "extra.json").open("w", encoding="utf-8") as f:
            json.dump(result.extra, f, indent=2, default=default)

    return run_dir
