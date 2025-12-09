# core/results/repository.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from .models import BacktestResult, Trade, EquityPoint


FLOAT_DECIMALS = 4


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _serialize_enum(value: Any) -> Any:
    # For enums like Side.BUY -> "BUY"
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _fmt_float(value: Any, ndigits: int = FLOAT_DECIMALS) -> Any:
    """
    Format floats for CSV output with fixed decimals.
    Returns the original value for non-floats.
    """
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return value


def _to_serializable(obj: Any, ndigits: int = FLOAT_DECIMALS) -> Any:
    """
    Convert an object into JSON-serializable structures and
    round floats recursively to the given precision.
    """
    if is_dataclass(obj):
        obj = asdict(obj)

    # Enums
    if hasattr(obj, "value") and not isinstance(obj, (str, int, float, bool, dict, list, tuple)):
        try:
            return obj.value
        except Exception:
            pass

    # Datetime-like
    if hasattr(obj, "isoformat") and not isinstance(obj, (str, int, float, bool, dict, list, tuple)):
        try:
            return obj.isoformat()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {k: _to_serializable(v, ndigits) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v, ndigits) for v in obj]

    if isinstance(obj, float):
        return round(obj, ndigits)

    return obj


def save_backtest_result(result: BacktestResult, base_dir: Path | str = "results") -> Path:
    """
    Save a BacktestResult into:
      results/<timestamp>/
        - summary.json
        - trades.csv
        - equity_curve.csv
        - extra.json (optional)
    """
    base_dir = Path(base_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / ts
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

    summary_serializable = _to_serializable(summary, FLOAT_DECIMALS)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_serializable, f, indent=2)

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
            writer.writerow(
                {
                    "trade_id": t.id,
                    "symbol": t.symbol,
                    "side": _serialize_enum(t.side),
                    "size": _fmt_float(t.size),
                    "entry_price": _fmt_float(t.entry_price),
                    "entry_time": t.entry_time.isoformat(),
                    "exit_price": _fmt_float(t.exit_price),
                    "exit_time": t.exit_time.isoformat(),
                    "realized_pnl": _fmt_float(t.net_pnl),
                    "fee": _fmt_float(getattr(t, "fee", 0.0)),
                }
            )

    # ------------------------------------------------------------------
    # 3) Equity curve (CSV)
    # ------------------------------------------------------------------
    with (run_dir / "equity_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "equity"])
        for p in result.equity_curve:
            writer.writerow([p.timestamp.isoformat(), _fmt_float(p.equity)])

    # ------------------------------------------------------------------
    # 4) Extra data (JSON, if present)
    # ------------------------------------------------------------------
    if result.extra:
        extra_serializable = _to_serializable(result.extra, FLOAT_DECIMALS)

        with (run_dir / "extra.json").open("w", encoding="utf-8") as f:
            json.dump(extra_serializable, f, indent=2)

    return run_dir
