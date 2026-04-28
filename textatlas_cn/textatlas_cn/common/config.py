"""Tiny config loader merging YAML + environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _expand(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, root) for v in value]
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand(cfg, REPO_ROOT)
    cfg.setdefault("project_root", str(REPO_ROOT))
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a possibly-relative path against project_root."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = Path(cfg["project_root"]) / p
    p.mkdir(parents=True, exist_ok=True)
    return p
