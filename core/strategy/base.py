# core/strategy/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import Candle, AccountState, Order, OrderFilledEvent


class IStrategy(ABC):
    """
    Strategy interface for the backtest engine.

    The engine calls:
      - on_candle() for each new candle
      - on_order_filled() whenever an order is filled
    """

    @abstractmethod
    def on_candle(self, candle: Candle, account: AccountState) -> List[Order]:
        """
        Called once per candle.

        Should return zero or more new Orders to be submitted to the engine.
        The engine will handle risk, fills and position accounting.
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

    def on_candle(self, candle: Candle, account: AccountState) -> List[Order]:
        return []

    def on_order_filled(self, event: OrderFilledEvent) -> None:
        return None
