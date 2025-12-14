# core/engine_actions.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.models import Order


class EngineActionType(str, Enum):
    """High-level commands a strategy can send to the engine."""

    PLACE_ORDER = "PLACE_ORDER"
    GRID_EXIT = "GRID_EXIT"  # cancel open orders + flatten open positions


@dataclass(frozen=True)
class EngineAction:
    """
    A strategy-to-engine command.

    - PLACE_ORDER: engine should submit/store the given Order.
    - GRID_EXIT: engine should cancel open orders for symbol and flatten positions.
    """

    type: EngineActionType
    symbol: str
    order: Optional[Order] = None
    exit_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type == EngineActionType.PLACE_ORDER:
            if self.order is None:
                raise ValueError("PLACE_ORDER requires order")
            if self.exit_reason is not None:
                raise ValueError("PLACE_ORDER must not set exit_reason")
        elif self.type == EngineActionType.GRID_EXIT:
            if self.order is not None:
                raise ValueError("GRID_EXIT must not set order")
            if not self.exit_reason:
                raise ValueError("GRID_EXIT requires exit_reason")
        else:
            raise ValueError(f"Unknown EngineActionType: {self.type!r}")

    @staticmethod
    def place(order: Order) -> "EngineAction":
        return EngineAction(
            type=EngineActionType.PLACE_ORDER,
            symbol=order.symbol,
            order=order,
            exit_reason=None,
        )

    @staticmethod
    def grid_exit(symbol: str, exit_reason: str) -> "EngineAction":
        return EngineAction(
            type=EngineActionType.GRID_EXIT,
            symbol=symbol,
            order=None,
            exit_reason=exit_reason,
        )
