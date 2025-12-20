# infra/exchange/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


@dataclass(frozen=True)
class AssetBalance:
    asset: str
    free: float
    locked: float


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    tick_size: float | None
    step_size: float | None
    min_notional: float | None
    min_qty: float | None


class SpotExchange(Protocol):
    def connect(self) -> None: ...
    def ping(self) -> None: ...
    def get_balances(self) -> Dict[str, AssetBalance]: ...
    def get_symbol_filters(self, symbol: str) -> SymbolFilters: ...
