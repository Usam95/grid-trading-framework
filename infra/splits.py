# infra/splits.py
from __future__ import annotations

from typing import Tuple

import pandas as pd

from infra.config import DataSplitConfig


def split_train_forward(
    df: pd.DataFrame,
    split_cfg: DataSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time series DataFrame into train (backtest) and forward sets.

    - For mode='ratio', split by row count.
    - For mode='date', split by index (DatetimeIndex), with:
        train   = rows with index < split_date
        forward = rows with index >= split_date

    Returns
    -------
    (train_df, forward_df) : two non-empty DataFrames whose union
    covers the original range without overlap.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if split_cfg.mode == "ratio":
        n = len(df)
        idx = int(n * split_cfg.train_ratio)

        if idx <= 0 or idx >= n:
            raise ValueError(
                f"Invalid train_ratio {split_cfg.train_ratio} for n={n}: "
                "must yield non-empty train and forward sets."
            )

        train = df.iloc[:idx].copy()
        forward = df.iloc[idx:].copy()

    elif split_cfg.mode == "date":
        if split_cfg.split_date is None:
            raise ValueError("split_date must be set when mode='date'.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "Date-based split requires DataFrame with a DatetimeIndex."
            )

        # Half-open interval to avoid overlap:
        #   train   = index < split_date
        #   forward = index >= split_date
        mask_train = df.index < split_cfg.split_date
        mask_forward = df.index >= split_cfg.split_date

        train = df.loc[mask_train].copy()
        forward = df.loc[mask_forward].copy()

        if train.empty or forward.empty:
            raise ValueError(
                f"Date split at {split_cfg.split_date} produced an empty "
                f"{'train' if train.empty else 'forward'} set."
            )
    else:
        raise ValueError(f"Unknown split mode: {split_cfg.mode!r}")

    return train, forward
