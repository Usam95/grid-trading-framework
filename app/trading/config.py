# app/trading/config.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from infra.config import RunConfig, RunMode
from infra.config_loader import load_run_config
from infra.config.trading_config import TradingRuntimeConfig


@dataclass(frozen=True)
class TradingSettings:
    run_cfg: RunConfig
    raw_cfg: Dict[str, Any]
    config_path: str

    execute: bool
    require_private_endpoints: bool

    trading: TradingRuntimeConfig


def _load_raw(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("PyYAML not installed but YAML config provided.")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def load_trading_settings(path: Union[str, Path]) -> TradingSettings:
    p = Path(path)
    raw = _load_raw(p)
    run_cfg = load_run_config(str(p))

    if run_cfg.mode not in (RunMode.PAPER, RunMode.LIVE):
        raise ValueError(f"Trading runtime requires mode paper/live, got: {run_cfg.mode}")

    if run_cfg.trading is None:
        run_cfg.trading = TradingRuntimeConfig()

    trading = run_cfg.trading
    execute = bool(trading.execute)
    require_private = bool(trading.require_private_endpoints)

    return TradingSettings(
        run_cfg=run_cfg,
        raw_cfg=raw,
        config_path=str(p),
        execute=execute,
        require_private_endpoints=require_private,
        trading=trading,
    )


def resolve_use_testnet_ws(run_cfg: RunConfig) -> bool:
    v = getattr(run_cfg.data, "use_testnet_ws", None)
    if v is None:
        return run_cfg.mode == RunMode.PAPER
    return bool(v)
