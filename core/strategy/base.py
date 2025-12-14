# core/strategy/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import Candle, AccountState, Order, OrderFilledEvent
from core.engine_actions import EngineAction

class IStrategy(ABC):
    """
    Strategy interface for the backtest engine.

    The engine calls:
      - on_candle() for each new candle
      - on_order_filled() whenever an order is filled
    """

    @abstractmethod
    def on_candle(self, candle: Candle, account: AccountState) -> List[EngineAction]:
        """
        Called once per candle.

    	Should return zero or more EngineActions for the engine to process.
        """
        raise NotImplementedError

    @abstractmethod
    def on_order_filled(self, event: OrderFilledEvent) -> None:
        """
        Called by the engine whenever an order (or part of it) is filled.
        Strategy can update its internal state (e.g. grid levels) here.
        """
        raise NotImplementedError


class BaseStrategy(IStrategy):
    """
    Optional base class with default no-op implementations.
    """

    def on_candle(self, candle: Candle, account: AccountState) -> List[EngineAction]:
        return []

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        return None
