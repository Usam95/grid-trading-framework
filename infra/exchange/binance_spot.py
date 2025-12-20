# infra/exchange/binance_spot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from binance.client import Client

from infra.config.engine_config import RunMode
from infra.exchange.base import AssetBalance, SymbolFilters, SpotExchange
from infra.secrets import EnvSecretsProvider


BINANCE_SPOT_TESTNET_REST = "https://testnet.binance.vision/api"
BINANCE_SPOT_PROD_REST = "https://api.binance.com/api"


@dataclass(frozen=True)
class BinanceSpotCredentials:
    api_key: str
    api_secret: str


class BinanceSpotExchange(SpotExchange):
    """
    Read-only Binance Spot adapter (Phase B1).

    - PAPER => Spot Testnet
    - LIVE  => Spot Prod

    Requires API key/secret to call signed endpoints like /api/v3/account (balances).
    """

    def __init__(
        self,
        *,
        mode: RunMode,
        creds: BinanceSpotCredentials,
        logger=None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.mode = mode
        self.creds = creds
        self.logger = logger
        self.timeout_seconds = float(timeout_seconds)
        self.client: Optional[Client] = None

    @classmethod
    def from_env(cls, *, mode: RunMode, secrets: EnvSecretsProvider, logger=None) -> "BinanceSpotExchange":
        if mode == RunMode.PAPER:
            # Prefer dedicated testnet env vars, fallback to generic names
            api_key = secrets.get("BINANCE_TESTNET_API_KEY") or secrets.require("BINANCE_API_KEY")
            api_secret = secrets.get("BINANCE_TESTNET_API_SECRET") or secrets.require("BINANCE_API_SECRET")
        else:
            api_key = secrets.require("BINANCE_API_KEY")
            api_secret = secrets.require("BINANCE_API_SECRET")

        return cls(
            mode=mode,
            creds=BinanceSpotCredentials(api_key=api_key, api_secret=api_secret),
            logger=logger,
        )

    def connect(self) -> None:
        # python-binance supports testnet=True, but we also set API_URL explicitly for robustness
        testnet = self.mode == RunMode.PAPER
        self.client = Client(
            api_key=self.creds.api_key,
            api_secret=self.creds.api_secret,
            tld="com",
            testnet=testnet,
            requests_params={"timeout": self.timeout_seconds},
        )

        # Force correct spot REST base (helps with some library versions)
        if testnet:
            self.client.API_URL = BINANCE_SPOT_TESTNET_REST
        else:
            self.client.API_URL = BINANCE_SPOT_PROD_REST

        self._log("info", "BinanceSpotExchange connected (mode=%s, api_url=%s)", self.mode.value, self.client.API_URL)

    def ping(self) -> None:
        if not self.client:
            raise RuntimeError("BinanceSpotExchange not connected.")
        _ = self.client.ping()
        self._log("info", "Binance ping OK.")

        # server time is handy for debugging
        st = self.client.get_server_time()
        self._log("info", "Binance server time: %s", st)

    def get_balances(self) -> Dict[str, AssetBalance]:
        if not self.client:
            raise RuntimeError("BinanceSpotExchange not connected.")
        account = self.client.get_account()

        out: Dict[str, AssetBalance] = {}
        for b in account.get("balances", []):
            asset = b.get("asset")
            free = float(b.get("free", 0.0))
            locked = float(b.get("locked", 0.0))
            out[asset] = AssetBalance(asset=asset, free=free, locked=locked)
        return out

    def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        if not self.client:
            raise RuntimeError("BinanceSpotExchange not connected.")
        info = self.client.get_exchange_info()

        sym = None
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                sym = s
                break
        if sym is None:
            raise ValueError(f"Symbol not found in exchangeInfo: {symbol}")

        tick_size = step_size = min_notional = min_qty = None

        for f in sym.get("filters", []):
            ftype = f.get("filterType")
            if ftype == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", "0") or 0.0)
            elif ftype == "LOT_SIZE":
                step_size = float(f.get("stepSize", "0") or 0.0)
                min_qty = float(f.get("minQty", "0") or 0.0)
            elif ftype == "MIN_NOTIONAL":
                # some symbols use "minNotional"
                mn = f.get("minNotional")
                if mn is not None:
                    min_notional = float(mn)

        return SymbolFilters(
            symbol=symbol,
            tick_size=tick_size,
            step_size=step_size,
            min_notional=min_notional,
            min_qty=min_qty,
        )

    def _log(self, level: str, msg: str, *args) -> None:
        if not self.logger:
            return
        fn = getattr(self.logger, level, None)
        if callable(fn):
            fn(msg, *args)
