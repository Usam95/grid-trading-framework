# core/results/live_repository.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LiveRunPaths:
    root: Path
    session_json: Path
    orders_csv: Path
    fills_csv: Path
    equity_csv: Path
    pnl_csv: Path
    events_jsonl: Path


class LiveRepository:
    def __init__(
        self,
        base_dir: str = "runs/live",
        run_name: str = "live_run",
        run_id: Optional[str] = None,
    ) -> None:
        ts = _utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        folder = run_id if run_id else f"{run_name}_{ts}"

        root = Path(base_dir) / folder
        root.mkdir(parents=True, exist_ok=True)

        self.paths = LiveRunPaths(
            root=root,
            session_json=root / "session.json",
            orders_csv=root / "orders.csv",
            fills_csv=root / "fills.csv",
            equity_csv=root / "equity.csv",
            pnl_csv=root / "pnl.csv",
            events_jsonl=root / "events.jsonl",
        )

        self._events_lock = Lock()
        self._csv_lock = Lock()
        self._init_csvs()

        self.paths.events_jsonl.touch(exist_ok=True)

    def _init_csvs(self) -> None:
        if not self.paths.orders_csv.exists():
            with self.paths.orders_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts", "symbol", "side", "type", "client_order_id", "exchange_order_id", "status", "qty", "price"]
                )

        if not self.paths.fills_csv.exists():
            with self.paths.fills_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts", "symbol", "side", "type", "client_order_id", "exchange_order_id",
                     "fill_qty", "fill_price", "cum_qty", "cum_quote"]
                )

        if not self.paths.equity_csv.exists():
            with self.paths.equity_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts", "symbol", "price", "base_free", "base_locked", "quote_free", "quote_locked", "equity_quote"]
                )

        if not self.paths.pnl_csv.exists():
            with self.paths.pnl_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        "ts", "symbol", "price",
                        "cash_quote", "base_position",
                        "invested_cost_quote",
                        "fees_paid_quote",
                        "realized_pnl_quote", "unrealized_pnl_quote", "total_pnl_quote",
                        "ledger_equity_quote",
                        "buyhold_equity_quote", "buyhold_return_pct",
                        "reason",
                    ]
                )

    def write_session(self, payload: Dict[str, Any]) -> None:
        self.paths.session_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_event(self, event: str, payload: Any) -> None:
        with self._events_lock:
            with self.paths.events_jsonl.open("a", encoding="utf-8") as f:
                rec = {"ts": _utcnow().isoformat(), "event": event, "payload": self._jsonify(payload)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def append_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        client_order_id: str,
        exchange_order_id: str,
        status: str,
        qty: float,
        price: float,
    ) -> None:
        with self._csv_lock:
            with self.paths.orders_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        _utcnow().isoformat(),
                        symbol,
                        side,
                        order_type,
                        client_order_id,
                        exchange_order_id,
                        status,
                        float(qty),
                        float(price),
                    ]
                )

    def append_fill(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        client_order_id: str,
        exchange_order_id: str,
        fill_qty: float,
        fill_price: float,
        cum_qty: float,
        cum_quote: float,
    ) -> None:
        with self._csv_lock:
            with self.paths.fills_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        _utcnow().isoformat(),
                        symbol,
                        side,
                        order_type,
                        client_order_id,
                        exchange_order_id,
                        float(fill_qty),
                        float(fill_price),
                        float(cum_qty),
                        float(cum_quote),
                    ]
                )

    def append_equity(
        self,
        *,
        symbol: str,
        price: float,
        base_free: float,
        base_locked: float,
        quote_free: float,
        quote_locked: float,
    ) -> None:
        equity_quote = float(quote_free + quote_locked) + float(base_free + base_locked) * float(price)
        with self._csv_lock:
            with self.paths.equity_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        _utcnow().isoformat(),
                        symbol,
                        float(price),
                        float(base_free),
                        float(base_locked),
                        float(quote_free),
                        float(quote_locked),
                        float(equity_quote),
                    ]
                )

    def append_pnl(
        self,
        *,
        symbol: str,
        price: float,
        cash_quote: float,
        base_position: float,
        invested_cost_quote: float,
        fees_paid_quote: float,
        realized_pnl_quote: float,
        unrealized_pnl_quote: float,
        total_pnl_quote: float,
        ledger_equity_quote: float,
        buyhold_equity_quote: float,
        buyhold_return_pct: float,
        reason: str,
    ) -> None:
        with self._csv_lock:
            with self.paths.pnl_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        _utcnow().isoformat(),
                        symbol,
                        float(price),
                        float(cash_quote),
                        float(base_position),
                        float(invested_cost_quote),
                        float(fees_paid_quote),
                        float(realized_pnl_quote),
                        float(unrealized_pnl_quote),
                        float(total_pnl_quote),
                        float(ledger_equity_quote),
                        float(buyhold_equity_quote),
                        float(buyhold_return_pct),
                        str(reason),
                    ]
                )

    def _jsonify(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (list, tuple)):
            return [self._jsonify(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._jsonify(v) for k, v in obj.items()}
        return obj
