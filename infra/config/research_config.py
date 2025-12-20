from __future__ import annotations

from typing import List, Optional, Literal

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


class ResearchSaveConfig(BaseModel):
    """
    Controls which full backtest artifacts are written to disk by the research app.

    - mode="best": write best_train + best_forward only
    - mode="topk": additionally write the top_k best train runs under output/<symbol>/topk/
    """

    mode: Literal["best", "topk"] = Field(
        default="best",
        description="Saving strategy for research runs.",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        description="How many top train runs to persist when mode='topk'.",
    )

    write_extra: bool = Field(
        default=True,
        description="Whether to write extra.json for each saved backtest result.",
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

    save: ResearchSaveConfig = Field(
        default_factory=ResearchSaveConfig,
        description="Persist best/topK backtest artifacts to disk (app-level behavior).",
    )

    primary_metric: str = Field(
    default="total_return_pct",
    description="Metric name used to rank parameter combinations."
    )

        # NEW
    n_jobs: int = Field(
        default=1,
        ge=1,
        description="Number of worker processes used for train-backtests.",
    )

    chunk_size: Optional[int] = Field(
        default=None,
        description="How many parameter combinations each worker evaluates per task. If None, a heuristic is used.",
    )

    mp_start_method: str = Field(
        default="spawn",
        description="Multiprocessing start method (Windows: 'spawn' only).",
    )

    progress_every: int = Field(
        default=15,
        ge=1,
        description="Log progress every N completed train runs.",
    )

