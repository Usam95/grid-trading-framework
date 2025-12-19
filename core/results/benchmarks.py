#core/results/benchmarks.py

from __future__ import annotations

from typing import Any, Dict

from core.models import Candle


def compute_buy_and_hold_benchmark(
    *,
    initial_quote: float,
    initial_base: float,
    fee_pct: float,
    candle0: Candle,
    candle_last: Candle,
) -> Dict[str, Any]:
    """
    Buy&Hold benchmark:
      - at candle0.close: buy base with ALL quote once (fee applied)
      - hold base until candle_last.close
      - report final equity in quote

    Assumes fee is paid in quote on notional:
      total_cost = price * qty * (1 + fee_pct)
      qty = quote / (price * (1 + fee_pct))
    """
    start_price = float(candle0.close)
    end_price = float(candle_last.close)

    if start_price <= 0.0:
        return {
            "enabled": True,
            "error": "start_price<=0",
        }

    q0 = float(initial_quote)
    b0 = float(initial_base)
    f = max(0.0, float(fee_pct))

    bought_qty = q0 / (start_price * (1.0 + f)) if q0 > 0 else 0.0
    base_total = b0 + bought_qty

    initial_equity = base_total * start_price
    final_equity = base_total * end_price

    ret_pct = 0.0
    if initial_equity > 0.0:
        ret_pct = (final_equity / initial_equity - 1.0) * 100.0

    return {
        "enabled": True,
        "start_price": start_price,
        "end_price": end_price,
        "initial_quote": q0,
        "initial_base": b0,
        "fee_pct": f,
        "bought_qty": bought_qty,
        "base_total": base_total,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": ret_pct,
        }
