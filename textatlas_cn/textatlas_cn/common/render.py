"""Text rendering utilities (Chinese-aware).

Includes:
- white-background pure text rendering with optional rotation/justification
- bbox text rendering on top of a real image (rectangular)
- irregular quadrilateral rendering via perspective transform
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema import BBox, FontAttrs, OcrLine


# ----------------------------------------------------------------------
# Pure-text rendering on white background (CleanTextSynth-CN).
# ----------------------------------------------------------------------

@dataclass
class CleanTextRenderConfig:
    canvas_size: tuple[int, int] = (1024, 1024)
    bg_color: tuple[int, int, int] = (255, 255, 255)
    margin: int = 60
    font_size_range: tuple[int, int] = (24, 64)
    rotation_range: tuple[float, float] = (-8.0, 8.0)
    line_spacing_range: tuple[float, float] = (1.05, 1.6)
    alignments: tuple[str, ...] = ("left", "center", "right")


def render_clean_text(
    text: str,
    font_path: str | Path,
    cfg: CleanTextRenderConfig | None = None,
    rng: random.Random | None = None,
    text_color: tuple[int, int, int] | None = None,
) -> tuple[Image.Image, FontAttrs, list[BBox]]:
    """Render Chinese text on a white canvas, returning image, font attrs, per-line bboxes."""
    cfg = cfg or CleanTextRenderConfig()
    rng = rng or random.Random()
    font_size = rng.randint(*cfg.font_size_range)
    font = ImageFont.truetype(str(font_path), font_size)
    color = text_color or (rng.randint(0, 80),) * 3
    rotation = rng.uniform(*cfg.rotation_range)
    spacing = rng.uniform(*cfg.line_spacing_range)
    alignment = rng.choice(cfg.alignments)

    canvas = Image.new("RGB", cfg.canvas_size, cfg.bg_color)
    draw = ImageDraw.Draw(canvas)

    max_w = cfg.canvas_size[0] - 2 * cfg.margin
    lines = _wrap_chinese(text, font, max_w, draw)
    line_height = int(font_size * spacing)
    total_h = line_height * len(lines)
    if total_h > cfg.canvas_size[1] - 2 * cfg.margin:
        # truncate vertically
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


def _wrap_chinese(text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """Break Chinese text into lines that fit within max_w. Honors original paragraph breaks."""
    out: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            out.append("")
            continue
        line = ""
        for ch in para:
            tentative = line + ch
            if draw.textlength(tentative, font=font) <= max_w:
                line = tentative
            else:
                out.append(line)
                line = ch
        if line:
            out.append(line)
    return out


# ----------------------------------------------------------------------
# Rectangular bbox rendering (StyledTextSynth / TextScenesHQ).
# ----------------------------------------------------------------------
def render_text_in_rect(
    image: Image.Image,
    bbox: BBox,
    text: str,
    font_path: str | Path,
    color: tuple[int, int, int] = (0, 0, 0),
    padding: float = 0.05,
) -> tuple[Image.Image, FontAttrs]:
    xs = [p[0] for p in bbox.points]
    ys = [p[1] for p in bbox.points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w, h = x1 - x0, y1 - y0
    pad_w, pad_h = int(w * padding), int(h * padding)
    rect_w, rect_h = max(int(w - 2 * pad_w), 1), max(int(h - 2 * pad_h), 1)

    font_size = _fit_chinese_font(font_path, text, rect_w, rect_h)
    font = ImageFont.truetype(str(font_path), font_size)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    lines = _wrap_chinese(text, font, rect_w, draw)
    line_h = int(font_size * 1.15)
    cy = y0 + pad_h + (rect_h - line_h * len(lines)) // 2
    for line in lines:
        line_w = draw.textlength(line, font=font)
        cx = x0 + pad_w + (rect_w - line_w) / 2
        draw.text((cx, cy), line, fill=color, font=font)
        cy += line_h

    attrs = FontAttrs(family=Path(font_path).stem, size=font_size, color=color, alignment="center")
    return overlay, attrs


def _fit_chinese_font(font_path: str | Path, text: str, w: int, h: int, low: int = 12, high: int = 200) -> int:
    """Binary-search the largest font size whose wrapped layout fits (w, h)."""
    while low < high:
        mid = (low + high + 1) // 2
        font = ImageFont.truetype(str(font_path), mid)
        # estimate fit
        dummy = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(dummy)
        lines = _wrap_chinese(text, font, w, draw)
        line_h = int(mid * 1.15)
        if line_h * len(lines) <= h:
            low = mid
        else:
            high = mid - 1
    return max(low, 12)


# ----------------------------------------------------------------------
# Irregular quadrilateral rendering via perspective transform.
# ----------------------------------------------------------------------
def render_text_in_quad(
    image: Image.Image,
    quad: BBox,
    text: str,
    font_path: str | Path,
    color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, FontAttrs]:
    import cv2

    pts = np.array(quad.points, dtype=np.float32)
    w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
    flat = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    flat, attrs = render_text_in_rect(flat.convert("RGB"), BBox.from_xyxy(0, 0, w, h), text, font_path, color)
    flat_rgba = flat.convert("RGBA")
    flat_arr = np.array(flat_rgba)

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, pts)
    img_arr = np.array(image.convert("RGBA"))
    warped = cv2.warpPerspective(flat_arr, M, (img_arr.shape[1], img_arr.shape[0]), flags=cv2.INTER_CUBIC)
    alpha = warped[..., 3:4] / 255.0
    img_arr[..., :3] = img_arr[..., :3] * (1 - alpha) + warped[..., :3] * alpha
    out = Image.fromarray(img_arr.astype(np.uint8)).convert("RGB")
    return out, attrs
