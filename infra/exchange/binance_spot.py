# infra/exchange/binance_spot.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

from binance.client import Client

from infra.config.engine_config import RunMode
from infra.logging_setup import get_logger


@dataclass(frozen=True)
class SymbolFilters:
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: Optional[float]


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _quantize_down_decimal(value: float, step: float) -> Decimal:
    """
    Round DOWN to the nearest multiple of `step` (exchange step size).
    Returns a Decimal to avoid float precision artifacts.
    """
    if step <= 0:
        return Decimal(str(value))

    d_value = Decimal(str(value))
    d_step = Decimal(str(step))

    n_steps = (d_value / d_step).to_integral_value(rounding=ROUND_DOWN)
    q = (n_steps * d_step)

    # quantize to same exponent as step (0.0001 -> 4 decimals)
    q = q.quantize(d_step, rounding=ROUND_DOWN)
    return q


def _fmt_decimal(d: Decimal) -> str:
    """
    Binance accepts string. Format Decimal without scientific notation and
    without trailing zeros (but preserving correct precision).
    """
    s = format(d, "f")  # never scientific
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


class BinanceSpotExchange:
    """
    Minimal Spot exchange adapter for:
      - PAPER: Binance Spot Testnet
      - LIVE:  Binance Spot mainnet
    """

    def __init__(
        self,
        mode: RunMode,
        api_key: str,
        api_secret: str,
        logger_name: str = "live.exchange",
    ) -> None:
        self.mode = mode
        self.logger = get_logger(logger_name)

        self.is_testnet = mode == RunMode.PAPER

        # python-binance client
        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=self.is_testnet)

        # Explicit and predictable REST base URLs
        if self.mode == RunMode.PAPER:
            self.api_url = "https://testnet.binance.vision/api"
        else:
            self.api_url = "https://api.binance.com/api"

        # IMPORTANT: enforce it on python-binance client too
        # (some versions don't switch API_URL reliably via testnet=True)
        self.client.API_URL = self.api_url

        self.logger.info(
            "BinanceSpotExchange connected (mode=%s, client.API_URL=%s)",
            mode.value,
            getattr(self.client, "API_URL", "<unknown>"),
        )

    # -------------------------
    # Read-only “health” calls
    # -------------------------
    def ping(self) -> None:
        self.client.ping()
        self.logger.info("Binance ping OK.")

    def server_time(self) -> Dict[str, Any]:
        st = self.client.get_server_time()
        self.logger.info("Binance server time: %s", st)
        return st

    def get_balances(self, non_zero_only: bool = True) -> Dict[str, Tuple[float, float]]:
        """
        Returns {asset: (free, locked)}.
        """
        acct = self.client.get_account()
        out: Dict[str, Tuple[float, float]] = {}

        for b in acct.get("balances", []):
            asset = b.get("asset")
            free = _to_float(b.get("free"))
            locked = _to_float(b.get("locked"))
            if non_zero_only and free == 0.0 and locked == 0.0:
                continue
            out[asset] = (free, locked)

        return out

    def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        info = self.client.get_symbol_info(symbol)
        if not info:
            raise ValueError(f"Symbol not found in exchange info: {symbol}")

        tick_size = 0.0
        step_size = 0.0
        min_qty = 0.0
        min_notional: Optional[float] = None

        for f in info.get("filters", []):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                tick_size = _to_float(f.get("tickSize"))
            elif ft == "LOT_SIZE":
                step_size = _to_float(f.get("stepSize"))
                min_qty = _to_float(f.get("minQty"))
            elif ft == "MIN_NOTIONAL":
                mn = f.get("minNotional")
                if mn is not None:
                    min_notional = _to_float(mn)

        return SymbolFilters(
            tick_size=tick_size,
            step_size=step_size,
            min_qty=min_qty,
            min_notional=min_notional,
        )

    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        B2: read-only open orders.
        """
        return self.client.get_open_orders(symbol=symbol)

    def get_last_price(self, symbol: str) -> float:
        """
        Public ticker price (used for B5 offset pricing).
        """
        res = self.client.get_symbol_ticker(symbol=symbol)
        if isinstance(res, dict) and "price" in res:
            return float(res["price"])
        raise RuntimeError(f"Unexpected ticker response: {res!r}")

    # -------------------------
    # Execution (B3 starts here)
    # -------------------------
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        client_order_id: Optional[str] = None,
        new_order_resp_type: str = "FULL",
        filters: Optional[SymbolFilters] = None,
    ) -> Dict[str, Any]:
        """
        Places a MARKET order (testnet in PAPER mode).
        """
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        if not _is_finite(quantity) or quantity <= 0:
            raise ValueError(f"quantity must be > 0 (got {quantity})")

        filt = filters or self.get_symbol_filters(symbol)

        adj_qty = _quantize_down_decimal(quantity, filt.step_size)
        min_qty_d = Decimal(str(filt.min_qty))

        if adj_qty < min_qty_d:
            raise ValueError(
                f"Quantity too small after step-size rounding: qty={quantity} -> {adj_qty}, min_qty={filt.min_qty}"
            )

        params: Dict[str, Any] = dict(
            symbol=symbol,
            side=side_u,
            type=Client.ORDER_TYPE_MARKET,
            quantity=_fmt_decimal(adj_qty),
            newOrderRespType=new_order_resp_type,
        )
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        self.logger.info(
            "Placing MARKET order: %s %s qty=%s (raw=%s)",
            side_u,
            symbol,
            params["quantity"],
            quantity,
        )

        return self.client.create_order(**params)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        *,
        client_order_id: Optional[str] = None,
        time_in_force: str = "GTC",
        new_order_resp_type: str = "RESULT",
        filters: Optional[SymbolFilters] = None,
    ) -> Dict[str, Any]:
        """
        B5: Places a LIMIT order (testnet in PAPER mode).

        - qty rounded DOWN by step_size
        - price rounded DOWN by tick_size
        - checks min_qty + (optional) min_notional
        """
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        if not _is_finite(quantity) or quantity <= 0:
            raise ValueError(f"quantity must be > 0 (got {quantity})")
        if not _is_finite(price) or price <= 0:
            raise ValueError(f"price must be > 0 (got {price})")

        filt = filters or self.get_symbol_filters(symbol)

        adj_qty = _quantize_down_decimal(quantity, filt.step_size)
        adj_px = _quantize_down_decimal(price, filt.tick_size)

        min_qty_d = Decimal(str(filt.min_qty))

        if adj_qty < min_qty_d:
            raise ValueError(
                f"Quantity too small after rounding: qty={quantity} -> {adj_qty}, min_qty={filt.min_qty}"
            )
        if adj_px <= Decimal("0"):
            raise ValueError(f"Price too small after rounding: price={price} -> {adj_px}")

        if filt.min_notional is not None:
            min_notional_d = Decimal(str(filt.min_notional))
            notional = adj_qty * adj_px
            if notional < min_notional_d:
                raise ValueError(
                    f"Notional too small: qty*price={notional} < min_notional={filt.min_notional} "
                    f"(qty={adj_qty}, price={adj_px})"
                )

        params: Dict[str, Any] = dict(
            symbol=symbol,
            side=side_u,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=time_in_force,
            quantity=_fmt_decimal(adj_qty),
            price=_fmt_decimal(adj_px),
            newOrderRespType=new_order_resp_type,
        )
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        self.logger.info(
            "Placing LIMIT order: %s %s qty=%s price=%s (raw_qty=%s raw_price=%s)",
            side_u,
            symbol,
            params["quantity"],
            params["price"],
            quantity,
            price,
        )

        return self.client.create_order(**params)

    def cancel_order(
        self,
        symbol: str,
        *,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        B5: Cancel an order by exchange orderId or clientOrderId.
        """
        if order_id is None and client_order_id is None:
            raise ValueError("Provide order_id or client_order_id")

        params: Dict[str, Any] = dict(symbol=symbol)
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id

        self.logger.info("Cancel order: %s", params)
        return self.client.cancel_order(**params)

    def cancel_all_open_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Convenience: cancel all open orders for symbol.
        Useful for shutdown / kill switch.
        """
        self.logger.info("Cancel ALL open orders for %s", symbol)
        return self.client.cancel_open_orders(symbol=symbol)

    def get_order(
        self,
        symbol: str,
        *,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if order_id is not None:
            return self.client.get_order(symbol=symbol, orderId=order_id)
        if client_order_id is not None:
            return self.client.get_order(symbol=symbol, origClientOrderId=client_order_id)
        raise ValueError("Provide order_id or client_order_id")

    # -------------------------
    # User Data Stream helpers
    # -------------------------
    def start_user_data_stream(self) -> str:
        """
        Create a listenKey for User Data Stream.

        python-binance return type differs by version:
        - some versions return {"listenKey": "..."}
        - some versions return "..." (raw string)
        """
        res = self.client.stream_get_listen_key()

        if isinstance(res, str) and res.strip():
            return res.strip()

        if isinstance(res, dict):
            lk = res.get("listenKey")
            if isinstance(lk, str) and lk.strip():
                return lk.strip()

        raise RuntimeError(f"Failed to acquire listenKey (unexpected response): {type(res)} value={res!r}")

    def keepalive_user_data_stream(self, listen_key: str) -> None:
        if not listen_key:
            raise ValueError("listen_key is empty")
        self.client.stream_keepalive(listen_key)

    def close_user_data_stream(self, listen_key: str) -> None:
        if not listen_key:
            return
        try:
            self.client.stream_close(listen_key)
        except Exception:
            pass

    def user_stream_ws_base(self) -> str:
        if self.mode == RunMode.PAPER:
            return "wss://stream.testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"

    def wait_until_terminal(
        self,
        symbol: str,
        *,
        order_id: int,
        timeout_sec: int = 30,
        poll_sec: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Simple polling (safe MVP) to observe fill result without user-data stream.
        Terminal statuses: FILLED, CANCELED, REJECTED, EXPIRED
        """
        end = time.time() + timeout_sec
        last: Dict[str, Any] = {}

        while time.time() < end:
            last = self.get_order(symbol, order_id=order_id)
            status = (last.get("status") or "").upper()

            if status in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
                return last

            time.sleep(poll_sec)

        return last
