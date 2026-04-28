from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from .fonts import FontCatalog
from .schema import DatasetSample, ImageBlock, TextBlock
from .text_utils import normalize_text, stable_id, truncate_by_units


@dataclass(frozen=True)
class InterleavedConfig:
    page_width: int = 1024
    page_height: int = 1024
    margin: int = 48
    min_font_size: int = 18
    max_font_size: int = 30
    max_text_units_per_box: int = 80


@dataclass(frozen=True)
class InterleavedItem:
    kind: str
    text: str | None = None
    image_path: str | None = None
    caption: str | None = None


@dataclass(frozen=True)
class InterleavedDocument:
    doc_id: str
    items: list[InterleavedItem]
    title: str | None = None
    source: str | None = None


def _fitz_rect(bbox: tuple[float, float, float, float]) -> fitz.Rect:
    return fitz.Rect(*bbox)


def _load_documents(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        if Path(path).suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
        return data if isinstance(data, list) else [data]


def _normalise_document(raw: dict[str, Any] | InterleavedDocument) -> dict[str, Any]:
    """Accept WIT/OBELICS-like simplified JSON and normalize to sections."""

    if isinstance(raw, InterleavedDocument):
        sections: list[dict[str, Any]] = []
        if raw.title:
            sections.append({"text": raw.title, "kind": "title"})
        for item in raw.items:
            section: dict[str, Any] = {"kind": item.kind}
            if item.text:
                section["text"] = item.text
            if item.image_path:
                section["image"] = item.image_path
            if item.caption:
                section["caption"] = item.caption
            if section:
                sections.append(section)
        return {"id": raw.doc_id, "sections": sections, "source": raw.source}

    sections = raw.get("sections")
    if sections:
        return {"id": str(raw.get("id", stable_id(json.dumps(raw, ensure_ascii=False)))), "sections": sections}

    texts = raw.get("texts") or raw.get("text_segments") or []
    images = raw.get("images") or []
    if isinstance(texts, str):
        texts = [texts]
    normalized_sections: list[dict[str, Any]] = []
    max_len = max(len(texts), len(images), 1)
    for index in range(max_len):
        section: dict[str, Any] = {}
        if index < len(texts):
            section["text"] = texts[index]
        if index < len(images):
            image_entry = images[index]
            section["image"] = image_entry.get("path") if isinstance(image_entry, dict) else image_entry
            if isinstance(image_entry, dict):
                section["caption"] = image_entry.get("caption")
        if section:
            normalized_sections.append(section)
    return {"id": str(raw.get("id", stable_id(json.dumps(raw, ensure_ascii=False)))), "sections": normalized_sections}


def build_interleaved_sample(
    raw_document: dict[str, Any] | InterleavedDocument,
    output_dir: str | Path,
    font_catalog: FontCatalog | None = None,
    config: InterleavedConfig | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> DatasetSample:
    """Generate a TextVisionBlend-style parseable PDF, rendered image, and JSON annotation."""

    config = config or InterleavedConfig()
    rng = rng or random.Random(seed)
    document = _normalise_document(raw_document)
    output_dir = Path(output_dir)
    pdf_dir = output_dir / "pdf"
    image_dir = output_dir / "images"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    sample_id = stable_id(document["id"], json.dumps(document["sections"], ensure_ascii=False), prefix="interleave_zh_")
    pdf_path = pdf_dir / f"{sample_id}.pdf"
    image_path = image_dir / f"{sample_id}.png"

    doc = fitz.open()
    page = doc.new_page(width=config.page_width, height=config.page_height)
    cursor_y = float(config.margin)
    text_blocks: list[TextBlock] = []
    image_blocks: list[ImageBlock] = []
    order = 0
    font_size = rng.randint(config.min_font_size, config.max_font_size)
    font_name = "china-s"
    font_path = font_catalog.random_font(rng) if font_catalog else None

    for section in document["sections"]:
        image_path_value = section.get("image")
        if image_path_value and Path(image_path_value).exists():
            with Image.open(image_path_value) as source_image:
                source_width, source_height = source_image.size
            max_image_width = config.page_width - 2 * config.margin
            max_image_height = config.page_height * 0.35
            scale = min(max_image_width / source_width, max_image_height / source_height, 1.0)
            width = source_width * scale
            height = source_height * scale
            if cursor_y + height > config.page_height - config.margin:
                break
            x0 = config.margin + rng.random() * max(1, max_image_width - width)
            rect = fitz.Rect(x0, cursor_y, x0 + width, cursor_y + height)
            page.insert_image(rect, filename=str(image_path_value))
            image_blocks.append(
                ImageBlock(
                    path=str(image_path_value),
                    bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                    caption=section.get("caption"),
                    reading_order=order,
                )
            )
            order += 1
            cursor_y = rect.y1 + 20

        text = normalize_text(str(section.get("text", "")))
        if text:
            text = truncate_by_units(text, config.max_text_units_per_box)
            box_height = max(80, min(220, (config.page_height - config.margin - cursor_y)))
            if cursor_y + box_height > config.page_height - config.margin:
                break
            rect = fitz.Rect(config.margin, cursor_y, config.page_width - config.margin, cursor_y + box_height)
            html = f"<div style='font-size:{font_size}px; line-height:1.35; font-family:sans-serif;'>{text}</div>"
            try:
                page.insert_htmlbox(rect, html)
            except Exception:
                page.insert_textbox(rect, text, fontsize=font_size, fontname=font_name)
            text_blocks.append(
                TextBlock(
                    text=text,
                    bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                    font=str(font_path) if font_path else font_name,
                    font_size=float(font_size),
                    color=(0, 0, 0),
                    reading_order=order,
                )
            )
            order += 1
            cursor_y = rect.y1 + 20

    doc.save(pdf_path)
    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    pix.save(image_path)
    doc.close()

    prompt_parts = ["生成一张中文图文混排页面，白色背景，保持以下阅读顺序："]
    for block in sorted([*text_blocks, *image_blocks], key=lambda item: item.reading_order or 0):
        if isinstance(block, TextBlock):
            prompt_parts.append(f"文字：{block.text}")
        else:
            prompt_parts.append(f"图片：{block.caption or '与正文相关的配图'}")
    return DatasetSample(
        sample_id=sample_id,
        subset="TextVisionBlendZH",
        image_path=str(image_path),
        prompt="\n".join(prompt_parts),
        text_blocks=text_blocks,
        image_blocks=image_blocks,
        metadata={"pdf_path": str(pdf_path), "source_id": document["id"]},
    )


def build_interleaved_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    font_catalog: FontCatalog | None = None,
    config: InterleavedConfig | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> list[DatasetSample]:
    rng = random.Random(seed)
    samples: list[DatasetSample] = []
    for raw in _load_documents(input_path)[:limit]:
        samples.append(build_interleaved_sample(raw, output_dir, font_catalog, config, rng))
    return samples


def render_interleaved_documents(
    documents: list[InterleavedDocument],
    output_dir: str | Path,
    font_catalog: FontCatalog | None = None,
    page_width: int = 1024,
    page_height: int = 1024,
    seed: int = 0,
) -> list[DatasetSample]:
    config = InterleavedConfig(page_width=page_width, page_height=page_height)
    rng = random.Random(seed)
    return [build_interleaved_sample(document, output_dir, font_catalog, config, rng) for document in documents]
