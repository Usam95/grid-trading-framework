# app/trading/account_state.py
from __future__ import annotations

from typing import Optional

from core.models import AccountState, Candle


def build_account_state(
    *,
    candle: Candle,
    base_asset: str,
    quote_asset: str,
    exchange,
    equity_tracker=None,
) -> Optional[AccountState]:
    """
    Build AccountState for strategy evaluation.
    Prefers EquityTracker (single source of truth); falls back to REST balances.
    """
    # Preferred: EquityTracker
    if equity_tracker is not None:
        try:
            equity_tracker.refresh(last_price=float(candle.close))
            return equity_tracker.account_state()
        except Exception:
            pass

    # Fallback: REST balances
    try:
        balances = exchange.get_balances(non_zero_only=False)
        b_free, b_locked = balances.get(base_asset, (0.0, 0.0))
        q_free, q_locked = balances.get(quote_asset, (0.0, 0.0))

        last_px = float(candle.close)
        base_total = float(b_free) + float(b_locked)
        quote_total = float(q_free) + float(q_locked)
        total_value = quote_total + base_total * last_px

        return AccountState(
            quote_balance=quote_total,
            base_inventory=base_total,
            reserved_quote=float(q_locked),
            reserved_base=float(b_locked),
            est_equity=total_value,
            last_price=last_px,
        )
    except Exception:
        return None
