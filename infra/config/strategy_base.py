#backtester\infra\config\strategy_base.py
from __future__ import annotations

from pydantic import BaseModel


class StrategyConfigBase(BaseModel):
    """
    Base class for all strategy configuration objects.

    Concrete strategies (grid, DCA, etc.) should derive from this and
    define a 'kind' discriminator plus their specific fields.
    """

    kind: str

    class Config:
        extra = "forbid"  # prevent silent typos in YAML
