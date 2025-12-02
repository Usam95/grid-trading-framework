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
    timeframe : str
        Target timeframe string, e.g. "1m", "5m", "15m", "1h", "4h", "1d".
        The loader assumes the *raw* data is at 1m resolution and will
        resample down to the requested timeframe.
    """
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None
    timeframe: str = "1m"


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
        Load data for the given DatasetConfig and return a sliced (and optionally
        resampled) DataFrame.
        """
        data_path = self.find_file_for_symbol(cfg.symbol)
        self.log.info(
            "Loading data for symbol=%s from %s (requested: start=%s, end=%s, timeframe=%s)",
            cfg.symbol,
            data_path,
            cfg.start,
            cfg.end,
            cfg.timeframe,
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

        timeframe = getattr(cfg, "timeframe", "1m") or "1m"
        if timeframe != "1m":
            final_df = self._resample_timeframe(sliced, timeframe)
        else:
            final_df = sliced

        self.log.info(
            "Loaded %d raw rows for %s (%s → %s). Returned %d rows after "
            "slicing/resampling to timeframe=%s (%s → %s).",
            len(df),
            cfg.symbol,
            df.index[0],
            df.index[-1],
            len(final_df),
            timeframe,
            final_df.index[0],
            final_df.index[-1],
        )

        return final_df

    # ------------------------------------------------------------------
    # Resampling helpers
    # ------------------------------------------------------------------
    def _timeframe_to_pandas_rule(self, timeframe: str) -> str:
        """
        Convert a human-friendly timeframe string like '1m', '5m', '1h', '4h', '1d'
        into a pandas resample rule ('1T', '5T', '1H', '4H', '1D').
        """
        tf = timeframe.strip()
        if not tf:
            raise ValueError("Timeframe string must not be empty.")

        unit = tf[-1].lower()
        try:
            value = int(tf[:-1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Expected formats like '1m', '5m', '1h', '1d'."
            ) from exc

        if unit == "m":
            return f"{value}T"  # minutes
        if unit == "h":
            return f"{value}H"  # hours
        if unit == "d":
            return f"{value}D"  # days

        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Use 'Xm', 'Xh' or 'Xd', e.g. '5m', '1h', '1d'."
        )

    def _resample_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Downsample from 1m candles to a higher timeframe.

        Aggregation:
          - Open  = first
          - High  = max
          - Low   = min
          - Close = last
          - Volume = sum
        """
        rule = self._timeframe_to_pandas_rule(timeframe)

        ohlc = df[["Open", "High", "Low", "Close"]].resample(rule).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            }
        )
        vol = df["Volume"].resample(rule).sum()

        out = ohlc.copy()
        out["Volume"] = vol

        # Drop empty bins (no candles in that interval)
        out.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        if out.empty:
            raise ValueError(
                f"Resampling to timeframe '{timeframe}' produced an empty DataFrame."
            )

        self.log.info(
            "Resampled data to timeframe=%s: %d → %d rows.",
            timeframe,
            len(df),
            len(out),
        )
        return out



class InMemoryDataSource:
    """
    Simple data source that always returns the same pre-loaded DataFrame.

    Useful for research/parameter-search runs to avoid re-reading and
    resampling the same data from disk on every engine run.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("InMemoryDataSource received an empty DataFrame.")
        self.df = df
        self.log = get_logger("data.memory")

    def load(self, cfg: DatasetConfig) -> pd.DataFrame:
        self.log.info(
            "Using in-memory DataFrame for symbol=%s timeframe=%s (%d rows).",
            cfg.symbol,
            cfg.timeframe,
            len(self.df),
        )
        # We assume df is already sliced & resampled for this timeframe.
        return self.df



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
            timeframe=getattr(dataset_conf, "timeframe", "1m"),
        )
    else:
        cfg = DatasetConfig(
            symbol=getattr(dataset_conf, "symbol"),
            start=getattr(dataset_conf, "start_date", None),
            end=getattr(dataset_conf, "end_date", None),
            timeframe=getattr(dataset_conf, "timeframe", "1m"),
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
        timeframe="1m",
    )
    df = ds.load(cfg)

    print(df.head())
    print(len(df))  # number of candles at requested timeframe
