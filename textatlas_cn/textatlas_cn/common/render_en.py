"""English-side rendering helpers analogous to ``render.py``.

We implement word-aware wrapping and a binary-search font fitter.
For mixed punctuation/numbers in either language the existing ``render.py``
``_fit_chinese_font`` works for both, but we keep this module so subsets
can pick the language-appropriate wrapper.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schema import BBox, FontAttrs


@dataclass
class CleanTextEnRenderConfig:
    canvas_size: tuple[int, int] = (1024, 1024)
    bg_color: tuple[int, int, int] = (255, 255, 255)
    margin: int = 60
    font_size_range: tuple[int, int] = (24, 64)
    rotation_range: tuple[float, float] = (-8.0, 8.0)
    line_spacing_range: tuple[float, float] = (1.05, 1.6)
    alignments: tuple[str, ...] = ("left", "center", "right")


_TOKEN_RE = re.compile(r"\S+|\s+")


def _wrap_english(text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw) -> list[str]:
    out: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            out.append("")
            continue
        words = para.split(" ")
        line = ""
        for w in words:
            tentative = (line + (" " if line else "") + w)
            if draw.textlength(tentative, font=font) <= max_w:
                line = tentative
            else:
                if line:
                    out.append(line)
                line = w
        if line:
            out.append(line)
    return out


def render_clean_text_en(
    text: str,
    font_path: str | Path,
    cfg: CleanTextEnRenderConfig | None = None,
    rng: random.Random | None = None,
    text_color: tuple[int, int, int] | None = None,
    forced_font_size: int | None = None,
    forced_alignment: str | None = None,
    forced_rotation: float | None = None,
    forced_line_spacing: float | None = None,
) -> tuple[Image.Image, FontAttrs, list[BBox]]:
    cfg = cfg or CleanTextEnRenderConfig()
    rng = rng or random.Random()
    font_size = forced_font_size or rng.randint(*cfg.font_size_range)
    font = ImageFont.truetype(str(font_path), font_size)
    color = text_color or (rng.randint(0, 80),) * 3
    rotation = cfg.rotation_range[0] if forced_rotation is None else forced_rotation
    if forced_rotation is None:
        rotation = rng.uniform(*cfg.rotation_range)
    spacing = forced_line_spacing or rng.uniform(*cfg.line_spacing_range)
    alignment = forced_alignment or rng.choice(cfg.alignments)

    canvas = Image.new("RGB", cfg.canvas_size, cfg.bg_color)
    draw = ImageDraw.Draw(canvas)

    max_w = cfg.canvas_size[0] - 2 * cfg.margin
    lines = _wrap_english(text, font, max_w, draw)
    line_height = int(font_size * spacing)
    total_h = line_height * len(lines)
    if total_h > cfg.canvas_size[1] - 2 * cfg.margin:
        keep = (cfg.canvas_size[1] - 2 * cfg.margin) // line_height
        lines = lines[: max(keep, 1)]
        total_h = line_height * len(lines)

    y = cfg.margin + (cfg.canvas_size[1] - 2 * cfg.margin - total_h) // 2
    bboxes: list[BBox] = []
    for line in lines:
        line_w = draw.textlength(line, font=font)
        if alignment == "left":
            x = cfg.margin
        elif alignment == "right":
            x = cfg.canvas_size[0] - cfg.margin - line_w
        else:
            x = (cfg.canvas_size[0] - line_w) / 2
        draw.text((x, y), line, fill=color, font=font)
        bboxes.append(BBox.from_xyxy(x, y, x + line_w, y + font_size))
        y += line_height

    if abs(rotation) > 0.01:
        canvas = canvas.rotate(rotation, resample=Image.BICUBIC, fillcolor=cfg.bg_color, expand=False)
    return canvas, FontAttrs(family=Path(font_path).stem, size=font_size, color=color, rotation=rotation, alignment=alignment), bboxes
