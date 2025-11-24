# utilities/data_utils/binance_downloader.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

HIST_DATA_ROOT = Path(__file__).resolve().parents[1] / "historical_data"


class BinanceMarketDataClient:
    """
    Minimal HTTP-client for Binance public market data endpoints (spot or futures).

    Only implements the parts needed to download historical OHLCV data for backtesting.
    """

    def __init__(
        self,
        futures: bool = False,
        base_url: str = "https://api.binance.com",
        request_timeout: int = 10,
    ) -> None:
        self.futures = futures
        self.base_url = base_url
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------
    # Low-level HTTP helper
    # ------------------------------------------------------------------
    def _get(self, endpoint: str, params: dict) -> list:
        url = self.base_url + endpoint
        resp = requests.get(url, params=params, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Error {resp.status_code} from {url}: {resp.text}"
            )
        return resp.json()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def get_symbols(self) -> List[str]:
        """Return list of tradable symbols for spot or futures."""
        endpoint = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
        data = self._get(endpoint, params={})
        return [x["symbol"] for x in data["symbols"]]

    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> list:
        """
        Request a single "page" of klines (candles).
        Docs: https://binance-docs.github.io/apidocs/
        """
        endpoint = "/fapi/v1/klines" if self.futures else "/api/v3/klines"
        params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        return self._get(endpoint, params=params)


@dataclass
class DownloadConfig:
    """
    Configuration for downloading historical data for a symbol.

    Attributes
    ----------
    symbol : str
        Ticker symbol, e.g. "XRPUSDT".
    interval : str
        Candle interval, e.g. "1m", "5m", "1h", "1d" etc.
    start : datetime
        Inclusive start datetime.
    end : datetime
        Inclusive end datetime.
    limit : int
        Number of candles per request (Binance max: 1000 for spot).
    """
    symbol: str
    interval: str
    start: datetime
    end: datetime
    limit: int = 1000


class BinanceHistoricalDownloader:
    """
    Downloads historical OHLCV data from Binance and stores it under `historical_data`.

    Folder layout:
        HIST_DATA_ROOT / <SYMBOL> / <SYMBOL>.parquet.gzip
    """

    def __init__(
        self,
        client: BinanceMarketDataClient,
        root: Path | str = HIST_DATA_ROOT,
        sleep_between_requests: float = 0.05,
    ) -> None:
        self.client = client
        self.root = Path(root)
        self.sleep_between_requests = sleep_between_requests

    # ------------------------------------------------------------------
    # Core download logic
    # ------------------------------------------------------------------
    def download_to_dataframe(self, cfg: DownloadConfig) -> pd.DataFrame:
        """
        Download all OHLCV candles for the given config as a pandas DataFrame.
        """
        start_ms = int(cfg.start.timestamp() * 1000)
        end_ms = int(cfg.end.timestamp() * 1000)

        all_rows = []

        while start_ms <= end_ms:
            try:
                raw = self.client.get_klines(
                    symbol=cfg.symbol,
                    interval=cfg.interval,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    limit=cfg.limit,
                )
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                print(f"[WARN] Request error: {e}. Cooling down for 3 minutes...")
                time.sleep(3 * 60)
                continue

            if not raw:
                # No more data
                break

            # raw[i] layout (spot & futures): [ openTime, open, high, low, close, volume, closeTime, ... ]
            for c in raw:
                all_rows.append(
                    (
                        int(c[0]),        # open time (ms)
                        float(c[1]),      # open
                        float(c[2]),      # high
                        float(c[3]),      # low
                        float(c[4]),      # close
                        float(c[5]),      # volume
                    )
                )

            # Move start to the ms after the last openTime to avoid duplicates
            last_open_time_ms = int(raw[-1][0])
            start_ms = last_open_time_ms + 1

            time.sleep(self.sleep_between_requests)

        if not all_rows:
            raise RuntimeError(
                f"No candles returned for {cfg.symbol} in range {cfg.start} – {cfg.end}"
            )

        df = pd.DataFrame(
            all_rows,
            columns=["OpenTimeMs", "Open", "High", "Low", "Close", "Volume"],
        )

        # Convert ms -> datetime and set as index
        df["Date"] = pd.to_datetime(df["OpenTimeMs"], unit="ms")
        df.set_index("Date", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

        return df

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def _symbol_dir(self, symbol: str) -> Path:
        d = self.root / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _symbol_file(self, symbol: str) -> Path:
        return self._symbol_dir(symbol) / f"{symbol}.parquet.gzip"

    def save_dataframe(self, symbol: str, df: pd.DataFrame) -> Path:
        """
        Store the DataFrame in Parquet format under the standard path.
        """
        path = self._symbol_file(symbol)
        df.to_parquet(path, compression="gzip")
        print(f"[INFO] Saved {symbol} data to {path}")
        return path

    def download_and_store(self, cfg: DownloadConfig) -> Path:
        """
        Convenience method: download data for cfg and store it directly.
        """
        df = self.download_to_dataframe(cfg)
        return self.save_dataframe(cfg.symbol, df)

    def update_existing_file(
        self,
        symbol: str,
        interval: str,
        end: datetime,
        limit: int = 1000,
    ) -> Path:
        """
        Update an existing symbol file by downloading candles from the last timestamp to `end`.

        If the file doesn't exist, this is effectively a fresh download.
        """
        path = self._symbol_file(symbol)

        if path.is_file():
            # Load existing data and determine last datetime
            existing = pd.read_parquet(path)
            if not isinstance(existing.index, pd.DatetimeIndex):
                raise ValueError(f"Existing file {path} has no DatetimeIndex.")
            existing.sort_index(inplace=True)
            last_dt = existing.index[-1]
            start = last_dt + pd.Timedelta(milliseconds=1)
            print(f"[INFO] Updating {symbol} from {last_dt} to {end}")
        else:
            # No file yet: start from a default far in the past (the user should adjust)
            raise FileNotFoundError(
                f"No existing file for {symbol}. "
                f"Use `download_and_store` first for a full initial history."
            )

        cfg = DownloadConfig(
            symbol=symbol,
            interval=interval,
            start=start.to_pydatetime(),
            end=end,
            limit=limit,
        )

        new_data = self.download_to_dataframe(cfg)
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        return self.save_dataframe(symbol, combined)


# ----------------------------------------------------------------------
# Example CLI usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: download several spot symbols from 2018-01-01 until now
    client = BinanceMarketDataClient(futures=False)
    downloader = BinanceHistoricalDownloader(client)

    start = datetime(2025, 11, 1)
    end = datetime.now()

    symbols: Iterable[str] = ["XRPUSDT"]
    interval = "1m"

    for symbol in symbols:
        cfg = DownloadConfig(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
        )
        downloader.download_and_store(cfg)
