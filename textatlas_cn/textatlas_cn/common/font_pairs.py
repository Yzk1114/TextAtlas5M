"""Paired Chinese ↔ English fonts for visually consistent bilingual rendering.

Each pair shares a typographic style (sans/serif/script/display) so that the
two parallel images look stylistically consistent except for the script.
The list is intentionally conservative; expand by appending to
``configs/fonts_en.yaml`` and ``configs/fonts.yaml``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EN_FONTS_YAML = REPO_ROOT / "configs" / "fonts_en.yaml"


def load_en_registry() -> list[dict[str, Any]]:
    if not EN_FONTS_YAML.exists():
        return []
    return yaml.safe_load(EN_FONTS_YAML.read_text(encoding="utf-8")).get("fonts", [])


# (zh_family, en_family, shared_style)
DEFAULT_PAIRS: list[tuple[str, str, str]] = [
    ("黑体", "sans", "modern_sans"),
    ("宋体", "serif", "literary_serif"),
    ("楷体", "script", "calligraphy"),
    ("行书", "script", "calligraphy"),
    ("行楷", "script", "calligraphy"),
    ("艺术体", "display", "display"),
]


def sample_font_pair(
    fonts_root: Path,
    en_fonts_root: Path | None = None,
    rng: random.Random | None = None,
    style: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a (zh_font_entry, en_font_entry) tuple.

    Both entries contain ``path``, ``name`` and ``family`` so that the
    rendering pipeline can reuse the existing fields produced by
    :func:`textatlas_cn.common.fonts.sample_font`.
    """
    from .fonts import load_registry as load_zh_registry

    rng = rng or random
    en_fonts_root = en_fonts_root or fonts_root.parent / "fonts_en"

    pair = rng.choice(DEFAULT_PAIRS) if style is None else next(p for p in DEFAULT_PAIRS if p[2] == style)
    zh_family, en_style, shared_style = pair

    zh_candidates = [f for f in load_zh_registry() if f.get("family") == zh_family and (fonts_root / f["file"]).exists()]
    if not zh_candidates:
        # Fallback: any installed Chinese font.
        zh_candidates = [f for f in load_zh_registry() if (fonts_root / f["file"]).exists()]
    en_candidates = [f for f in load_en_registry() if f.get("family") == en_style and (en_fonts_root / f["file"]).exists()]
    if not en_candidates:
        en_candidates = [f for f in load_en_registry() if (en_fonts_root / f["file"]).exists()]

    if not zh_candidates or not en_candidates:
        raise FileNotFoundError(
            "Need at least one Chinese and one English font installed. "
            "Run `python -m textatlas_cn.scripts.prepare_fonts` and "
            "`python -m textatlas_cn.scripts.prepare_fonts_en`."
        )
    zh = rng.choice(zh_candidates)
    en = rng.choice(en_candidates)
    return (
        {"path": str(fonts_root / zh["file"]), "shared_style": shared_style, **zh},
        {"path": str(en_fonts_root / en["file"]), "shared_style": shared_style, **en},
    )
