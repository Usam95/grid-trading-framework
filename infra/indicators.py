# infra/indicators.py

from __future__ import annotations

import pandas as pd

from infra.config.data_config import IndicatorConfig


# ----------------------------------------------------------------------
# Helper functions: indicator column names
# ----------------------------------------------------------------------


def atr_key(period: int) -> str:
    """Column name for ATR with given period."""
    return f"ATR_{period}"


def rsi_key(period: int) -> str:
    """Column name for RSI with given period."""
    return f"RSI_{period}"


def ema_key(period: int, col: str = "Close") -> str:
    """Column name for EMA on a given column (default: Close)."""
    return f"EMA_{period}_{col.upper()}"


# ----------------------------------------------------------------------
# Per-indicator adders (mutate df in-place and also return it)
# ----------------------------------------------------------------------


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add an ATR column with the given period.

    Uses classic True Range and simple moving average over 'period'.
    Expects columns: 'High', 'Low', 'Close'.
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
    return df


def add_ema(df: pd.DataFrame, period: int, col: str = "Close") -> pd.DataFrame:
    """
    Add an EMA column with the given period on the given source column
    (default: 'Close').
    """
    name = ema_key(period, col)
    df[name] = df[col].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int, col: str = "Close") -> pd.DataFrame:
    """
    Add an RSI column with the given period on the given source column
    (default: 'Close').
    """
    name = rsi_key(period)
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df[name] = 100 - (100 / (1 + rs))
    return df


# ----------------------------------------------------------------------
# High-level enrichment
# ----------------------------------------------------------------------


def enrich_indicators(df: pd.DataFrame, cfg: IndicatorConfig | None) -> pd.DataFrame:
    """
    Compute all indicators configured in IndicatorConfig and attach them
    as new columns to the DataFrame.

    - ATR columns:  ATR_<period>
    - EMA columns:  EMA_<period>_<COL> (currently COL == 'CLOSE')
    - RSI columns:  RSI_<period>

    All computations are in-place; df is also returned for convenience.
    """
    if cfg is None:
        return df

    # ATRs
    if cfg.compute_atr:
        for period in cfg.atr_periods:
            add_atr(df, period)

    # RSIs
    if cfg.compute_rsi:
        for period in cfg.rsi_periods:
            add_rsi(df, period)

    # EMAs
    if cfg.compute_ema:
        for period in cfg.ema_periods:
            add_ema(df, period)

    return df
