# core/execution/constraints.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models import Order, Side
from infra.config.engine_config import ConstraintConfig, InsufficientFundsMode


@dataclass(frozen=True)
class ConstraintDecision:
    accept: bool
    reason: Optional[str] = None


class OrderConstraintPolicy:
    """
    Engine-level order feasibility and optional resizing.

    Works on AVAILABLE funds (i.e. after reservations if enabled).
    Modifies the passed order in-place when resizing is enabled.
    """

    def __init__(self, cfg: ConstraintConfig, fee_pct: float) -> None:
        self.cfg = cfg
        self.fee_pct = float(fee_pct)

    def evaluate(self, order: Order, *, available_quote: float, available_base: float) -> ConstraintDecision:
        if not self.cfg.enabled:
            return ConstraintDecision(True)

        if order.qty <= 0:
            return ConstraintDecision(False, "qty<=0")

        min_qty = float(self.cfg.min_order_qty or 0.0)

        if order.side == Side.BUY:
            if order.price is None or order.price <= 0:
                return ConstraintDecision(False, "buy_price<=0")

            required = float(order.price) * float(order.qty) * (1.0 + self.fee_pct)
            if required <= available_quote + 1e-12:
                return ConstraintDecision(order.qty >= min_qty, "qty_below_min" if order.qty < min_qty else None)

            if self.cfg.insufficient_funds_mode == InsufficientFundsMode.SKIP:
                return ConstraintDecision(False, "insufficient_quote_skip")

            denom = float(order.price) * (1.0 + self.fee_pct)
            new_qty = available_quote / denom if denom > 0 else 0.0
            if new_qty < min_qty:
                return ConstraintDecision(False, "resize_below_min_qty")

            order.qty = new_qty
            return ConstraintDecision(True, f"resized_buy_qty={new_qty:.8f}")

        # SELL
        required_base = float(order.qty)
        if required_base <= available_base + 1e-12:
            return ConstraintDecision(order.qty >= min_qty, "qty_below_min" if order.qty < min_qty else None)

        if self.cfg.insufficient_funds_mode == InsufficientFundsMode.SKIP:
            return ConstraintDecision(False, "insufficient_base_skip")

        new_qty = max(0.0, available_base)
        if new_qty < min_qty:
            return ConstraintDecision(False, "resize_below_min_qty")

        order.qty = new_qty
        return ConstraintDecision(True, f"resized_sell_qty={new_qty:.8f}")
