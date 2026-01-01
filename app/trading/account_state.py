# app/trading/account_state.py

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from core.models import AccountState, Candle
from core.live.equity_tracker import EquityTracker

logger = logging.getLogger("trade.account_state")


def build_account_state(
    *,
    candle: Candle,
    base_asset: str,
    quote_asset: str,
    exchange,
    equity_tracker: Optional[EquityTracker] = None,
    **_ignored,  # keeps it forward/backward compatible with other call sites
) -> Optional[AccountState]:
    """
    Build a minimal, strategy-ready AccountState for live/paper trading.

    - Prefer EquityTracker snapshot if available (cached balances + last price).
    - Otherwise fall back to exchange.get_balances().
    """
    try:
        last_px = float(candle.close)
        if last_px <= 0.0:
            raise ValueError(f"Invalid candle.close={candle.close!r}")

        ts = getattr(candle, "ts", None) or datetime.now(timezone.utc)

        # 1) Prefer equity tracker snapshot
        if equity_tracker is not None:
            px, quote_bal, base_bal = equity_tracker.snapshot(last_price=last_px)
            q_free, q_locked = float(quote_bal.free), float(quote_bal.locked)
            b_free, b_locked = float(base_bal.free), float(base_bal.locked)

            px_f = float(px)
            if px_f > 0.0:
                last_px = px_f
        else:
            # 2) Fallback: direct exchange balances
            bals = exchange.get_balances([quote_asset, base_asset])
            q_free, q_locked = map(float, bals.get(quote_asset, (0.0, 0.0)))
            b_free, b_locked = map(float, bals.get(base_asset, (0.0, 0.0)))

        base_total = b_free + b_locked
        quote_total = q_free + q_locked

        total_value = quote_total + base_total * last_px

        # Best-effort cost basis (true cost basis should come from FillLedger)
        invested_cost = base_total * last_px

        return AccountState(
            cash_balance=quote_total,
            total_value=total_value,
            invested_cost=invested_cost,
            base_free=b_free,
            base_locked=b_locked,
            quote_free=q_free,
            quote_locked=q_locked,
        )

    except Exception:
        logger.exception(
            "build_account_state failed (base=%s quote=%s close=%r ts=%r)",
            base_asset,
            quote_asset,
            getattr(candle, "close", None),
            getattr(candle, "ts", None),
        )
        return None
