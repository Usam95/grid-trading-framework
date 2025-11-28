# infra/data_source.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from infra.logging_setup import get_logger

# Root folder where all historical data is stored:
#   historical_data/<SYMBOL>/<SYMBOL>.parquet.gzip
# or
#   historical_data/<SYMBOL>/<SYMBOL>.csv
HIST_DATA_ROOT = Path(__file__).resolve().parents[1] / "historical_data"


@dataclass
class DatasetConfig:
    """
    Configuration object describing what data to load.

    Attributes
    ----------
    symbol : str
        Ticker symbol, e.g. "XRPUSDT".
    start : Optional[str]
        Inclusive start datetime (e.g. "2021-01-01" or "2021-01-01 00:00:00").
        If None, the earliest available candle is used.
    end : Optional[str]
        Inclusive end datetime.
        If None, the latest available candle is used.
    """
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None


class LocalFileDataSource:
    """
    Loads OHLCV data from local files for backtesting.

    - Files are expected under: root / <SYMBOL> / <SYMBOL>.parquet.gzip (preferred)
      or root / <SYMBOL> / <SYMBOL>.csv.
    - Data is returned as a pandas DataFrame with a DatetimeIndex and
      columns: ["Open", "High", "Low", "Close", "Volume"].
    """

    def __init__(self, root: Path | str = HIST_DATA_ROOT) -> None:
        self.root = Path(root)
        self.log = get_logger("data")

    # ------------------------------------------------------------------
    # Low-level file discovery
    # ------------------------------------------------------------------
    def find_file_for_symbol(self, symbol: str) -> Path:
        """
        Return the path to the data file for a given symbol.

        Preference order:
        1. <root>/<symbol>/<symbol>.parquet.gzip
        2. <root>/<symbol>/<symbol>.csv
        """
        symbol_dir = self.root / symbol
        self.log.debug(
            "Looking for data for symbol=%s under %s",
            symbol,
            symbol_dir,
        )

        if not symbol_dir.is_dir():
            msg = f"No folder found for symbol {symbol!r} under {self.root}"
            self.log.error(msg)
            raise FileNotFoundError(msg)

        parquet_candidate = symbol_dir / f"{symbol}.parquet.gzip"
        csv_candidate = symbol_dir / f"{symbol}.csv"

        if parquet_candidate.is_file():
            self.log.info("Using parquet data file for %s: %s", symbol, parquet_candidate)
            return parquet_candidate
        if csv_candidate.is_file():
            self.log.info("Using CSV data file for %s: %s", symbol, csv_candidate)
            return csv_candidate

        msg = (
            f"No data file found for symbol {symbol!r} in {symbol_dir}. "
            f"Expected {symbol}.parquet.gzip or {symbol}.csv"
        )
        self.log.error(msg)
        raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # High-level load entry point
    # ------------------------------------------------------------------
    def load(self, cfg: DatasetConfig) -> pd.DataFrame:
        """
        Load data for the given DatasetConfig and return a sliced DataFrame.
        """
        data_path = self.find_file_for_symbol(cfg.symbol)
        self.log.info(
            "Loading data for symbol=%s from %s (requested: start=%s, end=%s)",
            cfg.symbol,
            data_path,
            cfg.start,
            cfg.end,
        )

        # Determine file type by suffixes (.parquet.gzip vs .csv)
        suffixes = data_path.suffixes
        if ".parquet" in suffixes:
            df = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
        else:
            msg = f"Unsupported file format for {data_path}"
            self.log.error(msg)
            raise ValueError(msg)

        # Ensure DatetimeIndex and sorted index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
            else:
                msg = f"Data for {cfg.symbol} has no DatetimeIndex and no 'Date' column."
                self.log.error(msg)
                raise ValueError(msg)

        df.sort_index(inplace=True)

        # Normalize and apply start/end
        start = pd.to_datetime(cfg.start) if cfg.start else df.index[0]
        end = pd.to_datetime(cfg.end) if cfg.end else df.index[-1]

        # Clip to available range if needed
        if start < df.index[0]:
            start = df.index[0]
        if end > df.index[-1]:
            end = df.index[-1]

        sliced = df.loc[start:end]

        if sliced.empty:
            msg = (
                f"No data for {cfg.symbol} in requested range "
                f"({cfg.start=} – {cfg.end=}). Available range is "
                f"{df.index[0]} – {df.index[-1]}"
            )
            self.log.error(msg)
            raise ValueError(msg)

        self.log.info(
            "Loaded %d rows for %s. Available range=%s → %s. Returned slice=%s → %s (%d rows).",
            len(df),
            cfg.symbol,
            df.index[0],
            df.index[-1],
            sliced.index[0],
            sliced.index[-1],
            len(sliced),
        )

        return sliced


# ----------------------------------------------------------------------
# Backwards-compatible helpers (optional)
# ----------------------------------------------------------------------


def get_path(ticker: str) -> Path:
    """
    Backwards-compatible wrapper that mimics your old get_path(ticker) API
    but delegates to LocalFileDataSource.
    """
    ds = LocalFileDataSource()
    return ds.find_file_for_symbol(ticker)


def load_data(dataset_conf: DatasetConfig) -> pd.DataFrame:
    """
    Backwards-compatible wrapper for your old load_data(dataset_conf) function.

    NOTE: `dataset_conf` must have attributes:
        - symbol
        - start or start_date
        - end or end_date
    """
    # Allow both old & new names (start_date/end_date)
    if hasattr(dataset_conf, "start") or hasattr(dataset_conf, "end"):
        cfg = DatasetConfig(
            symbol=getattr(dataset_conf, "symbol"),
            start=getattr(dataset_conf, "start", None),
            end=getattr(dataset_conf, "end", None),
        )
    else:
        cfg = DatasetConfig(
            symbol=getattr(dataset_conf, "symbol"),
            start=getattr(dataset_conf, "start_date", None),
            end=getattr(dataset_conf, "end_date", None),
        )

    ds = LocalFileDataSource()
    return ds.load(cfg)


# ----------------------------------------------------------------------
# Example CLI usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ds = LocalFileDataSource()
    cfg = DatasetConfig(
        symbol="XRPUSDT",
        start="2021-11-1",
        end="2025-11-20",
    )
    df = ds.load(cfg)

    print(df.head())
    print(len(df))  # should be close to 1T (1 minute) or None but spaced by 1 min
