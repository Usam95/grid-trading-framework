# infra/marketdata/binance_user_stream.py
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
import websocket


try:
    # Most likely already present in your project (kline streamer style).
    from websocket import WebSocketApp
except ImportError as e:
    raise ImportError(
        "Missing dependency: websocket-client. Install it via: pip install websocket-client"
    ) from e


@dataclass
class UserStreamConfig:
    keepalive_interval_sec: int = 30 * 60          # Binance recommends keepalive periodically (<=60min expiry)
    reconnect_backoff_sec: int = 5
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10


ExecutionReportHandler = Callable[[dict[str, Any]], None]
GenericEventHandler = Callable[[dict[str, Any]], None]


class BinanceUserDataStream:
    """
    Binance User Data Stream:
      - starts listenKey via REST
      - connects ws to .../ws/<listenKey>
      - keepalive via REST
      - logs and forwards events
    """

    def __init__(
        self,
        exchange: Any,
        logger: Any,
        cfg: Optional[UserStreamConfig] = None,
        on_execution_report: Optional[ExecutionReportHandler] = None,
        on_account_position: Optional[GenericEventHandler] = None,
        on_balance_update: Optional[GenericEventHandler] = None,
        on_any_event: Optional[GenericEventHandler] = None,
    ) -> None:
        self.exchange = exchange
        self.log = logger
        self.cfg = cfg or UserStreamConfig()

        self.on_execution_report = on_execution_report
        self.on_account_position = on_account_position
        self.on_balance_update = on_balance_update
        self.on_any_event = on_any_event

        self._stop = threading.Event()
        self._listen_key: Optional[str] = None
        self._ws: Optional[WebSocketApp] = None

        self._ws_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None

    @property
    def listen_key(self) -> Optional[str]:
        return self._listen_key

    def start(self) -> None:
        self._stop.clear()
        try:
            self._listen_key = self.exchange.start_user_data_stream()
        except Exception as e:
            self.log.warning(f"Failed to start user data stream (listenKey): {e}")
            raise

        lk_short = self._listen_key[:8] + "..." if self._listen_key else None
        self.log.info(f"User data stream listenKey acquired: {lk_short}")

        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

    def stop(self) -> None:
        self._stop.set()

        # close ws (will unblock run_forever)
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

        # close listenKey
        try:
            if self._listen_key:
                self.exchange.close_user_data_stream(self._listen_key)
                self.log.info("User data stream closed (listenKey deleted).")
        except Exception as e:
            self.log.warning(f"Failed to close user data stream: {e}")

    # ---------------------------
    # internals
    # ---------------------------

    def _keepalive_loop(self) -> None:
        assert self._listen_key is not None
        while not self._stop.is_set():
            # Sleep in smaller chunks so shutdown is quick
            total = self.cfg.keepalive_interval_sec
            for _ in range(total):
                if self._stop.is_set():
                    return
                time.sleep(1)

            try:
                self.exchange.keepalive_user_data_stream(self._listen_key)
                self.log.info("User data stream keepalive OK.")
            except Exception as e:
                self.log.warning(f"User data stream keepalive failed: {e}")

    def _ws_loop(self) -> None:
        assert self._listen_key is not None
        ws_base = self.exchange.user_stream_ws_base()
        url = f"{ws_base}/{self._listen_key}"

        self.log.info(f"Connecting user websocket: {url[:60]}...")

        while not self._stop.is_set():
            self._ws = WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            try:
                self._ws.run_forever(
                    ping_interval=self.cfg.ws_ping_interval,
                    ping_timeout=self.cfg.ws_ping_timeout,
                )
            except Exception as e:
                self.log.warning(f"User websocket run_forever error: {e}")

            if self._stop.is_set():
                return

            self.log.warning(f"User websocket disconnected. Reconnecting in {self.cfg.reconnect_backoff_sec}s...")
            time.sleep(self.cfg.reconnect_backoff_sec)

    def _on_open(self, ws: Any) -> None:
        self.log.info("User websocket connected.")

    def _on_close(self, ws: Any, status_code: Any, msg: Any) -> None:
        self.log.warning(f"User websocket closed: status={status_code} msg={msg}")

    def _on_error(self, ws: Any, error: Any) -> None:
        self.log.warning(f"User websocket error: {error}")

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            evt = json.loads(message)
        except Exception:
            self.log.warning(f"User stream: could not parse message: {message[:200]}")
            return

        if self.on_any_event:
            try:
                self.on_any_event(evt)
            except Exception as e:
                self.log.warning(f"on_any_event handler error: {e}")

        event_type = evt.get("e")

        if event_type == "executionReport":
            self._log_execution_report(evt)
            if self.on_execution_report:
                try:
                    self.on_execution_report(evt)
                except Exception as e:
                    self.log.warning(f"on_execution_report handler error: {e}")

        elif event_type == "outboundAccountPosition":
            self._log_account_position(evt)
            if self.on_account_position:
                try:
                    self.on_account_position(evt)
                except Exception as e:
                    self.log.warning(f"on_account_position handler error: {e}")

        elif event_type == "balanceUpdate":
            self._log_balance_update(evt)
            if self.on_balance_update:
                try:
                    self.on_balance_update(evt)
                except Exception as e:
                    self.log.warning(f"on_balance_update handler error: {e}")

        else:
            # keep it light; you can switch this to debug if it gets noisy
            self.log.debug(f"User stream event: {event_type} keys={list(evt.keys())}")

    def _log_execution_report(self, e: dict[str, Any]) -> None:
        # Common fields:
        # s=symbol, S=side, o=orderType, X=orderStatus
        # i=orderId, c=clientOrderId
        # l=lastFilledQty, L=lastFilledPrice
        # z=cumFilledQty, Z=cumQuoteQty
        symbol = e.get("s")
        side = e.get("S")
        otype = e.get("o")
        status = e.get("X")
        order_id = e.get("i")
        client_id = e.get("c")

        last_qty = e.get("l")
        last_px = e.get("L")
        cum_qty = e.get("z")
        cum_quote = e.get("Z")

        self.log.info(
            "EXEC_REPORT "
            f"{symbol} {side} {otype} status={status} "
            f"orderId={order_id} clientId={client_id} "
            f"lastFill={last_qty}@{last_px} cumQty={cum_qty} cumQuote={cum_quote}"
        )

    def _log_account_position(self, e: dict[str, Any]) -> None:
        # B = list of balances in this event
        # each entry: a=asset, f=free, l=locked
        balances = e.get("B", [])
        # Keep it compact: only show assets you care about (XRP/USDT/BNB) by default
        interesting = []
        for b in balances:
            a = b.get("a")
            if a in {"XRP", "USDT", "BNB"}:
                interesting.append(f"{a} free={b.get('f')} locked={b.get('l')}")
        if interesting:
            self.log.info("ACCOUNT_POS " + " | ".join(interesting))
        else:
            self.log.debug(f"ACCOUNT_POS balances={len(balances)}")

    def _log_balance_update(self, e: dict[str, Any]) -> None:
        # a=asset, d=delta, T=clearTime
        self.log.info(
            f"BALANCE_UPDATE asset={e.get('a')} delta={e.get('d')} clearTime={e.get('T')}"
        )
