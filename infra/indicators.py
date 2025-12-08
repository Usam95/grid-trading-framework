# infra/indicators.py
from __future__ import annotations

from typing import Iterable

import pandas as pd

from infra.config.data_config import IndicatorConfig



# ---------------------------------------------------------------------------
# Key helpers – single source of truth for column names
# ---------------------------------------------------------------------------

def atr_key(period: int) -> str:
    return f"ATR_{period}"


def ema_key(period: int) -> str:
    return f"EMA_{period}"


def rsi_key(period: int) -> str:
    return f"RSI_{period}"


# ---------------------------------------------------------------------------
# Low-level indicator implementations
# ---------------------------------------------------------------------------

def _add_atr(df: pd.DataFrame, period: int) -> None:
    """
    Add an ATR_<period> column in-place.

    Assumes df has 'High', 'Low', 'Close' columns.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df[atr_key(period)] = tr.rolling(period, min_periods=period).mean()


def _add_ema(df: pd.DataFrame, period: int) -> None:
    """
    Add an EMA_<period> column in-place on Close.
    """
    df[ema_key(period)] = df["Close"].ewm(span=period, adjust=False).mean()


def _add_rsi(df: pd.DataFrame, period: int) -> None:
    """
    Add an RSI_<period> column in-place on Close.
    Classic Wilder-style RSI.
    """
    close = df["Close"]
    delta = close.diff()

    gain = (delta.clip(lower=0)).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()

    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    df[rsi_key(period)] = rsi


# ---------------------------------------------------------------------------
# High-level enrichment
# ---------------------------------------------------------------------------

def enrich_indicators(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """
    Compute all requested indicators in-place on df and return df.

    Uses the lists from IndicatorConfig:
      - atr_periods
      - ema_periods
      - rsi_periods

    If any list is empty or the corresponding `compute_*` flag is False,
    that indicator type is skipped.
    """

    # --- ATR ---
    if cfg.compute_atr:
        for period in cfg.atr_periods:
            _add_atr(df, period)

    # --- EMA ---
    if cfg.compute_ema:
        for period in cfg.ema_periods:
            _add_ema(df, period)

    # --- RSI ---
    if cfg.compute_rsi:
        for period in cfg.rsi_periods:
            _add_rsi(df, period)

    return df
