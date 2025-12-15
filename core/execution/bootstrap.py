# core/execution/bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from core.models import Candle, Position, PositionSide
from infra.config.engine_config import BacktestEngineConfig, BootstrapMode


@dataclass(frozen=True)
class BootstrapOutcome:
    started_at: datetime
    start_price: float

    initial_quote: float
    initial_base_qty: float

    purchased_base_qty: float
    spent_quote: float
    fees_paid: float

    mode: str


def bootstrap_portfolio(
    *,
    symbol: str,
    candle0: Candle,
    engine_cfg: BacktestEngineConfig,
    positions: Dict[str, Position],
    cash_balance_ref: Dict[str, float],
) -> Optional[BootstrapOutcome]:
    """
    Mutates:
      - positions (adds bootstrap positions)
      - cash_balance_ref["cash"] (deducts quote spent)

    Returns BootstrapOutcome (store into BacktestResult.extra).
    """
    price = float(candle0.close)
    if price <= 0:
        return None

    fee_pct = float(engine_cfg.trading_fee_pct or 0.0)

    initial_quote = float(cash_balance_ref["cash"])
    purchased_base = 0.0
    spent_quote = 0.0
    fees_paid = 0.0

    # 0) Pre-owned base inventory
    init_base = float(getattr(engine_cfg, "initial_base_qty", 0.0) or 0.0)
    if init_base > 0.0:
        pos = Position(
            id=str(uuid4()),
            symbol=symbol,
            side=PositionSide.LONG,
            entry_price=price,
            size=init_base,
            opened_at=candle0.timestamp,
            fees_paid=0.0,
        )
        positions[pos.id] = pos

    mode = engine_cfg.bootstrap.mode if getattr(engine_cfg, "bootstrap", None) else BootstrapMode.LONG_ONLY

    # 1) neutral_split
    if mode == BootstrapMode.NEUTRAL_SPLIT:
        pct = min(1.0, max(0.0, float(engine_cfg.bootstrap.initial_quote_to_base_pct or 0.0)))
        if pct > 0.0 and cash_balance_ref["cash"] > 0.0:
            quote_to_use = cash_balance_ref["cash"] * pct
            notional = quote_to_use / (1.0 + fee_pct)
            qty = notional / price
            fee = notional * fee_pct
            total_cost = notional + fee

            if total_cost <= cash_balance_ref["cash"] + 1e-12 and qty > 0:
                positions[str(uuid4())] = Position(
                    id=str(uuid4()),
                    symbol=symbol,
                    side=PositionSide.LONG,
                    entry_price=price,
                    size=qty,
                    opened_at=candle0.timestamp,
                    fees_paid=fee,
                )
                cash_balance_ref["cash"] -= total_cost
                purchased_base += qty
                spent_quote += total_cost
                fees_paid += fee

    # 2) neutral_topup
    elif mode == BootstrapMode.NEUTRAL_TOPUP:
        current_base = sum(p.size for p in positions.values() if p.is_open and p.symbol == symbol)
        desired_qty: Optional[float] = None

        if engine_cfg.bootstrap.target_base_qty is not None:
            desired_qty = float(engine_cfg.bootstrap.target_base_qty)
        elif engine_cfg.bootstrap.target_base_value_pct is not None:
            pct = min(1.0, max(0.0, float(engine_cfg.bootstrap.target_base_value_pct)))
            est_equity = cash_balance_ref["cash"] + current_base * price
            desired_qty = (est_equity * pct) / price

        if desired_qty is not None and desired_qty > current_base + 1e-12:
            missing_qty = desired_qty - current_base

            available_quote = float(cash_balance_ref["cash"])
            if engine_cfg.bootstrap.max_topup_quote is not None:
                available_quote = min(available_quote, float(engine_cfg.bootstrap.max_topup_quote))

            affordable_notional = available_quote / (1.0 + fee_pct)
            affordable_qty = affordable_notional / price
            qty = min(missing_qty, affordable_qty)

            if qty > 0:
                notional = qty * price
                fee = notional * fee_pct
                total_cost = notional + fee

                if total_cost <= cash_balance_ref["cash"] + 1e-12:
                    positions[str(uuid4())] = Position(
                        id=str(uuid4()),
                        symbol=symbol,
                        side=PositionSide.LONG,
                        entry_price=price,
                        size=qty,
                        opened_at=candle0.timestamp,
                        fees_paid=fee,
                    )
                    cash_balance_ref["cash"] -= total_cost
                    purchased_base += qty
                    spent_quote += total_cost
                    fees_paid += fee

    return BootstrapOutcome(
        started_at=candle0.timestamp,
        start_price=price,
        initial_quote=initial_quote,
        initial_base_qty=init_base,
        purchased_base_qty=purchased_base,
        spent_quote=spent_quote,
        fees_paid=fees_paid,
        mode=str(mode),
    )
