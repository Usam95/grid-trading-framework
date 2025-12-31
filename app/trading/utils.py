# app/trading/utils.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Tuple


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_secret(s: str, keep: int = 4) -> str:
    if not s:
        return "<EMPTY>"
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)


def split_symbol(symbol: str) -> Tuple[str, str]:
    """
    Best-effort split for spot symbols. Works for common quotes like USDT, USDC, FDUSD, EUR, BTC, ETH.
    """
    s = symbol.upper()
    for q in ("USDT", "USDC", "BUSD", "FDUSD", "EUR", "BTC", "ETH", "BNB", "TRY"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)], q
    return s[:-4], s[-4:]


def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def ms_to_dt_utc(ms: Any) -> datetime:
    ms_i = int(ms)
    return datetime.fromtimestamp(ms_i / 1000.0, tz=timezone.utc)
