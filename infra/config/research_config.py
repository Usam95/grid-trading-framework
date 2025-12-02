from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from core.strategy.grid_strategy_simple import GridSpacing, GridRangeMode


class GridParamGridConfig(BaseModel):
    """
    Parameter grid for the simple grid strategy.

    Each field is a list of candidate values. The research runner will
    take the Cartesian product of non-empty lists to generate individual
    strategy configs.

    Example YAML:

      research:
        enabled: true
        max_runs: 100

        param_grid:
          base_order_size: [5.0, 10.0, 20.0]
          n_levels: [10, 20, 30]
          lower_pct: [0.05, 0.10]
          upper_pct: [0.05, 0.10]
          range_mode: ["PERCENT"]
          spacing: ["ARITHMETIC"]
    """

    base_order_size: List[float] = Field(
        default_factory=list,
        description="Candidate values for base_order_size.",
    )
    n_levels: List[int] = Field(
        default_factory=list,
        description="Candidate values for n_levels.",
    )
    lower_pct: List[float] = Field(
        default_factory=list,
        description="Candidate values for lower_pct.",
    )
    upper_pct: List[float] = Field(
        default_factory=list,
        description="Candidate values for upper_pct.",
    )

    lower_price: List[float] = Field(
        default_factory=list,
        description="Candidate values for lower_price (absolute mode).",
    )
    upper_price: List[float] = Field(
        default_factory=list,
        description="Candidate values for upper_price (absolute mode).",
    )

    range_mode: List[GridRangeMode] = Field(
        default_factory=list,
        description="Candidate range modes, e.g. ['PERCENT', 'ABSOLUTE'].",
    )
    spacing: List[GridSpacing] = Field(
        default_factory=list,
        description="Candidate spacings, e.g. ['ARITHMETIC', 'GEOMETRIC'].",
    )

    def is_empty(self) -> bool:
        """
        Convenience helper so a research runner can quickly check if
        anything is actually configured.
        """
        return all(
            not getattr(self, field)
            for field in [
                "base_order_size",
                "n_levels",
                "lower_pct",
                "upper_pct",
                "lower_price",
                "upper_price",
                "range_mode",
                "spacing",
            ]
        )


class GridResearchConfig(BaseModel):
    """
    Top-level research configuration for grid strategy parameter search.

    If 'enabled' is False or param_grid.is_empty(), a research runner
    can simply no-op.
    """

    enabled: bool = Field(
        False,
        description="If false, research/param-search is effectively disabled.",
    )

    max_runs: Optional[int] = Field(
        default=None,
        description="Optional cap on number of parameter combinations to run.",
    )

    param_grid: GridParamGridConfig = Field(
        default_factory=GridParamGridConfig,
        description="Parameter grid for generating strategy configs.",
    )

    primary_metric: str = Field(
    default="total_return_pct",
    description="Metric name used to rank parameter combinations."
    )

