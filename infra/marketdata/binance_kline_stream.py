# infra/marketdata/binance_kline_stream.py
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import websockets

from core.models import Candle


DEFAULT_PROD_WS_BASE = "wss://stream.binance.com:9443/ws"
DEFAULT_TESTNET_WS_BASE = "wss://stream.testnet.binance.vision/ws"


@dataclass(frozen=True)
class StreamStatus:
    connected: bool
    last_message_ts_utc: Optional[datetime]
    url: str


class BinanceKlineStream:
    """
    Minimal Binance kline websocket stream that yields CLOSED candles only.

    - Runs an asyncio websocket consumer inside a background thread
    - Exposes a blocking `next_closed_candle()` for the main thread
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        *,
        use_testnet: bool,
        ws_base_url: Optional[str] = None,
        reconnect_delay_sec: float = 5.0,
        max_queue_size: int = 1000,
        logger=None,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.use_testnet = use_testnet
        self.ws_base_url = ws_base_url or (DEFAULT_TESTNET_WS_BASE if use_testnet else DEFAULT_PROD_WS_BASE)

        self.reconnect_delay_sec = reconnect_delay_sec
        self._q: "queue.Queue[Candle]" = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._status = StreamStatus(connected=False, last_message_ts_utc=None, url=self._stream_url())
        self._logger = logger

    def _stream_name(self) -> str:
        # Binance stream names use lowercase symbol
        return f"{self.symbol.lower()}@kline_{self.interval}"

    def _stream_url(self) -> str:
        return f"{self.ws_base_url}/{self._stream_name()}"

    def status(self) -> StreamStatus:
        return self._status

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, name="binance-kline-stream", daemon=True)
        self._thread.start()
        self._log("info", "BinanceKlineStream started: %s", self._stream_url())

    def stop(self) -> None:
        self._stop.set()
        self._log("info", "BinanceKlineStream stop requested.")
        if self._thread:
            self._thread.join(timeout=5)

    def next_closed_candle(self, timeout_sec: Optional[float] = None) -> Candle:
        """
        Blocking call that returns the next CLOSED candle.
        Raises TimeoutError if timeout elapses.
        """
        try:
            return self._q.get(timeout=timeout_sec)
        except queue.Empty as e:
            raise TimeoutError(f"No closed candle received within {timeout_sec}s") from e

    # -------------------------
    # Internal thread + asyncio
    # -------------------------

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_forever())
        finally:
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    async def _run_forever(self) -> None:
        url = self._stream_url()
        while not self._stop.is_set():
            try:
                self._status = StreamStatus(connected=False, last_message_ts_utc=self._status.last_message_ts_utc, url=url)
                self._log("info", "Connecting websocket: %s", url)

                async with websockets.connect(url, ping_interval=180, ping_timeout=600) as ws:
                    self._status = StreamStatus(connected=True, last_message_ts_utc=datetime.now(timezone.utc), url=url)
                    self._log("info", "Websocket connected: %s", url)

                    while not self._stop.is_set():
                        raw = await ws.recv()
                        self._status = StreamStatus(connected=True, last_message_ts_utc=datetime.now(timezone.utc), url=url)

                        candle = self._parse_closed_kline(raw)
                        if candle is not None:
                            self._push_candle(candle)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log("error", "Websocket error: %s", e)
                if self._stop.is_set():
                    break
                time.sleep(self.reconnect_delay_sec)

        self._log("info", "Websocket loop exited: %s", url)

    def _parse_closed_kline(self, raw: str) -> Optional[Candle]:
        """
        Accepts raw stream payload. Handles combined streams wrapper as well.
        Returns Candle only when kline is closed (x == True).
        """
        try:
            msg = json.loads(raw)
        except Exception:
            return None

        # Combined stream wrapper: {"stream": "...", "data": {...}}
        if isinstance(msg, dict) and "data" in msg and isinstance(msg["data"], dict):
            msg = msg["data"]

        if not isinstance(msg, dict):
            return None

        k = msg.get("k")
        if not isinstance(k, dict):
            return None

        is_closed = bool(k.get("x"))
        if not is_closed:
            return None

        # kline fields (strings)
        ts_ms = int(k["t"])
        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

        return Candle(
            timestamp=ts,
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )

    def _push_candle(self, candle: Candle) -> None:
        try:
            self._q.put_nowait(candle)
        except queue.Full:
            # Drop oldest by draining one item then retry once (best-effort)
            try:
                _ = self._q.get_nowait()
            except Exception:
                pass
            try:
                self._q.put_nowait(candle)
            except Exception:
                pass

    def _log(self, level: str, msg: str, *args) -> None:
        if self._logger is None:
            return
        fn = getattr(self._logger, level, None)
        if callable(fn):
            fn(msg, *args)
