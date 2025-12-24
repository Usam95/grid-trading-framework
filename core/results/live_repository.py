# core/results/live_repository.py
from __future__ import annotations

import csv
import json
import os
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
            events_jsonl=root / "events.jsonl",
        )

        self._events_lock = Lock()
        self._init_csvs()

        # ensure events file exists
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

    def write_session(self, payload: Dict[str, Any]) -> None:
        self.paths.session_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------------
    # Events (JSONL) - thread-safe
    # ---------------------------------------------------------------------
    def append_event(self, *args: Any) -> None:
        """
        Supports BOTH call styles:

        1) append_event({"type": "...", "payload": {...}})
        2) append_event("type.name", {...})

        But now it is thread-safe and writes exactly one line atomically.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            event = dict(args[0])
        elif len(args) == 2 and isinstance(args[0], str):
            event = {"type": args[0], "payload": args[1]}
        else:
            raise TypeError("append_event expects either (dict) or (event_type: str, payload: Any)")

        event.setdefault("ts", _utcnow().isoformat())
        line = json.dumps(event, default=self._json_default, ensure_ascii=False, separators=(",", ":")) + "\n"

        # Serialize concurrent writers (user stream thread + main loop)
        with self._events_lock:
            with self.paths.events_jsonl.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    # ---------------------------------------------------------------------
    # Equity
    # ---------------------------------------------------------------------
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
        equity = (quote_free + quote_locked) + (base_free + base_locked) * price
        with self.paths.equity_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    _utcnow().isoformat(),
                    symbol,
                    f"{price:.8f}",
                    f"{base_free:.8f}",
                    f"{base_locked:.8f}",
                    f"{quote_free:.8f}",
                    f"{quote_locked:.8f}",
                    f"{equity:.8f}",
                ]
            )
    @staticmethod
    def _json_default(o: Any) -> Any:
        """
        Make events JSONL robust:
          - dataclasses (e.g., Balance) -> dict
          - datetime -> iso string
          - sets -> list
          - fallback -> string
        """
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, set):
            return list(o)
        # last resort (never crash logging)
        return str(o)
    

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
        with self.paths.orders_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                _utcnow().isoformat(),
                symbol,
                side,
                order_type,
                client_order_id,
                exchange_order_id,
                status,
                f"{qty:.8f}",
                f"{price:.8f}",
            ])

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
        with self.paths.fills_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                _utcnow().isoformat(),
                symbol,
                side,
                order_type,
                client_order_id,
                exchange_order_id,
                f"{fill_qty:.8f}",
                f"{fill_price:.8f}",
                f"{cum_qty:.8f}",
                f"{cum_quote:.8f}",
            ])