from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .fonts import FontCatalog
from .schema import DatasetSample, TextBlock
from .text_utils import normalize_text, stable_id, truncate_by_units


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1024
    height: int = 1024
    margin: int = 64
    min_font_size: int = 24
    max_font_size: int = 56
    rotation_degrees: tuple[int, ...] = (-2, -1, 0, 1, 2)
    alignments: tuple[str, ...] = ("left", "center", "right")
    line_spacing: float = 1.25
    max_units: int | None = None


def _load_font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(font_path), size=size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap Chinese or mixed text without requiring whitespace tokenization."""

    probe = Image.new("RGB", (32, 32), "white")
    draw = ImageDraw.Draw(probe)
    paragraphs = [normalize_text(paragraph) for paragraph in text.replace("\r\n", "\n").split("\n")]
    lines: list[str] = []
    for paragraph in paragraphs:
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and _text_width(draw, candidate, font) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines or [""]


def render_text_image(
    text: str,
    output_path: str | Path,
    font_catalog: FontCatalog | None = None,
    config: RenderConfig | None = None,
    rng: random.Random | None = None,
    subset: str = "CleanTextSynthZH",
) -> DatasetSample:
    """Render a CleanTextSynth-style Chinese long-text image on a white canvas."""

    config = config or RenderConfig()
    rng = rng or random.Random()
    if config.max_units is not None:
        text = truncate_by_units(text, config.max_units)
    text = normalize_text(text)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font_path = font_catalog.random_font(rng) if font_catalog else None
    font_size = rng.randint(config.min_font_size, config.max_font_size)
    font = _load_font(font_path, font_size)
    max_width = config.width - 2 * config.margin
    lines = wrap_text(text, font, max_width)

    line_height = math.ceil(font_size * config.line_spacing)
    max_lines = max(1, (config.height - 2 * config.margin) // line_height)
    lines = lines[:max_lines]
    text_height = len(lines) * line_height
    y = max(config.margin, (config.height - text_height) // 2)

    color = tuple(rng.randint(0, 70) for _ in range(3))
    image = Image.new("RGB", (config.width, config.height), "white")
    draw = ImageDraw.Draw(image)
    alignment = rng.choice(config.alignments)
    bboxes: list[tuple[float, float, float, float]] = []
    for line in lines:
        line_width = _text_width(draw, line, font)
        if alignment == "center":
            x = (config.width - line_width) / 2
        elif alignment == "right":
            x = config.width - config.margin - line_width
        else:
            x = config.margin
        draw.text((x, y), line, font=font, fill=color)
        bboxes.append((x, y, x + line_width, y + line_height))
        y += line_height

    rotation = rng.choice(config.rotation_degrees)
    if rotation:
        image = image.rotate(rotation, expand=False, fillcolor="white")
    image.save(output_path)

    if bboxes:
        x0 = min(box[0] for box in bboxes)
        y0 = min(box[1] for box in bboxes)
        x1 = max(box[2] for box in bboxes)
        y1 = max(box[3] for box in bboxes)
    else:
        x0, y0, x1, y1 = config.margin, config.margin, config.width - config.margin, config.height - config.margin

    sample_id = stable_id(subset, text, str(output_path), prefix="clean_zh_")
    prompt = f"生成一张白色背景的中文长文本图片，图片中清晰排版以下文字：{text}"
    return DatasetSample(
        sample_id=sample_id,
        subset=subset,
        image_path=str(output_path),
        prompt=prompt,
        text_blocks=[
            TextBlock(
                text=text,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                font=str(font_path) if font_path else None,
                font_size=float(font_size),
                color=color,
                reading_order=0,
            )
        ],
        metadata={"rotation_degrees": rotation, "alignment": alignment},
    )


def render_many_clean_text(
    texts: Iterable[str],
    output_dir: str | Path,
    font_catalog: FontCatalog | None = None,
    config: RenderConfig | None = None,
    seed: int = 0,
) -> list[DatasetSample]:
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    samples: list[DatasetSample] = []
    for index, text in enumerate(texts):
        sample_id = stable_id(str(index), text, prefix="clean_zh_")
        output_path = output_dir / f"{sample_id}.png"
        samples.append(render_text_image(text, output_path, font_catalog, config, rng))
    return samples
