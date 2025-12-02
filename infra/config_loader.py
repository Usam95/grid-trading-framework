# infra/config_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from infra.config import RunConfig

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - only hit if pyyaml is missing
    yaml = None


def _load_raw(path: Path) -> dict:
    """
    Load a raw dict from a YAML or JSON file.
    """
    text = path.read_text(encoding="utf-8")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "pyyaml is not installed but a YAML config file was provided."
            )
        return yaml.safe_load(text) or {}

    if suffix == ".json":
        return json.loads(text)

    raise ValueError(f"Unsupported config format: {path} (use .yaml/.yml or .json)")


def load_run_config(path: Union[str, Path]) -> RunConfig:
    """
    Load a RunConfig from the given YAML/JSON file.

    Example:
        cfg = load_run_config('config/grid_run.yml')
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw = _load_raw(p)
    try:
        return RunConfig.parse_obj(raw)
    except ValidationError as exc:  # pragma: no cover - mostly developer-time failure
        # You can pretty-print here if you like
        raise ValueError(f"Config validation failed for {p}:\n{exc}") from exc
