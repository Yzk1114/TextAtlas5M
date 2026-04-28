"""Chinese font registry and loaders.

`prepare_fonts.py` uses :func:`load_registry` and :func:`download_all`.
Subset pipelines call :func:`sample_font` to obtain a random font path.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FONTS_YAML = REPO_ROOT / "configs" / "fonts.yaml"


def load_registry(yaml_path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(yaml_path or FONTS_YAML)
    return yaml.safe_load(p.read_text(encoding="utf-8")).get("fonts", [])


def font_path(entry: dict[str, Any], fonts_root: Path) -> Path:
    return fonts_root / entry["file"]


def sample_font(
    fonts_root: Path,
    family: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    rng = rng or random
    candidates = load_registry()
    if family:
        candidates = [f for f in candidates if f.get("family") == family]
    available = [c for c in candidates if (fonts_root / c["file"]).exists()]
    if not available:
        raise FileNotFoundError(
            "No fonts available. Run `python -m textatlas_cn.scripts.prepare_fonts` first."
        )
    chosen = rng.choice(available)
    return {"path": str(fonts_root / chosen["file"]), **chosen}
