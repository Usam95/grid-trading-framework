# infra/secrets.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from infra.config.engine_config import RunMode


@dataclass(frozen=True)
class SecretRef:
    name: str
    required: bool = True


class EnvSecretsProvider:
    """
    Read secrets from environment variables.
    Works for local dev + later Azure Container Instance env vars.

    Expected env vars:
      PAPER (Spot Testnet):
        BINANCE_TESTNET_API_KEY
        BINANCE_TESTNET_API_SECRET

      LIVE (Mainnet):
        BINANCE_API_KEY
        BINANCE_API_SECRET
    """

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(name, default)

    def require(self, name: str) -> str:
        v = os.environ.get(name)
        if not v:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return v

    def get_binance_keys(self, mode: RunMode) -> Tuple[str, str]:
        """
        Returns (api_key, api_secret) depending on mode.
        PAPER => Spot Testnet keys
        LIVE  => mainnet keys
        """
        if mode == RunMode.PAPER:
            key = self.require("BINANCE_TESTNET_API_KEY")
            secret = self.require("BINANCE_TESTNET_API_SECRET")
            return key, secret

        # backtest shouldn't call this, but if it does, default to mainnet names
        key = self.require("BINANCE_API_KEY")
        secret = self.require("BINANCE_API_SECRET")
        return key, secret
