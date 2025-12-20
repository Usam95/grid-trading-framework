# infra/secrets.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SecretRef:
    name: str
    required: bool = True


class EnvSecretsProvider:
    """
    Read secrets from environment variables.
    Works for local dev + later Azure Container Instance env vars.
    """

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(name, default)

    def require(self, name: str) -> str:
        v = os.environ.get(name)
        if not v:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return v
