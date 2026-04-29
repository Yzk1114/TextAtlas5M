"""Bilingual / parallel TextVisionBlend.

Inputs are parallel ``{type, images, sections|texts_zh|texts_en, id}`` records
where every text block already has both languages; you can pre-build such
records from Wikipedia-zh + Wikipedia-en pairs (linked by Wikidata QID) or by
LLM-translating one side.

We render two PDFs that share *every* layout box position and image, so the
only visual difference is the text glyphs themselves.
"""
from __future__ import annotations

import argparse
import io
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.font_pairs import sample_font_pair
from ...common.io import save_image, stable_id
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.schema import BBox, OcrLine, TextAtlasSample
from .layout import PageLayout, plan_obelics, plan_wit


CAPTION_PROMPT_ZH = "请用不超过50字的中文为这张图像生成简洁的描述，不要照抄图中文字。"
CAPTION_PROMPT_EN = "Generate a concise English caption (<=50 words) for the image; do not copy any in-image text."


def _render_layout(layout: PageLayout, font_path: str, language: str) -> tuple[bytes, list[dict[str, Any]]]:
    doc = fitz.open()
    page = doc.new_page(width=layout.page_w, height=layout.page_h)
    page.insert_font(fontname=language, fontfile=font_path)
    annotations: list[dict[str, Any]] = []
    for box in layout.boxes:
        rect = fitz.Rect(box.x, box.y, box.x + box.w, box.y + box.h)
        if box.type == "text" and box.payload:
            html = (
                f'<div style="font-family:{language};font-size:14pt;line-height:1.4;'
                f'color:#202020;text-align:justify;">{box.payload}</div>'
            )
            page.insert_htmlbox(rect, html)
            annotations.append({"type": "text", "role": box.role, "bbox": [box.x, box.y, box.x + box.w, box.y + box.h], "text": box.payload, "font": language})
        elif box.type == "image" and box.payload:
            try:
                img_bytes = Path(box.payload).read_bytes() if not box.payload.startswith("http") else __import__("requests").get(box.payload, timeout=20).content
                page.insert_image(rect, stream=img_bytes, keep_proportion=True)
                annotations.append({"type": "image", "role": box.role, "bbox": [box.x, box.y, box.x + box.w, box.y + box.h], "src": box.payload})
            except Exception:
                continue
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes, annotations


def _twin_layout(layout: PageLayout, texts_other: list[str]) -> PageLayout:
    """Produce a layout with identical box geometry but text payloads replaced
    by ``texts_other`` in the same iteration order. Image boxes are untouched."""
    twin = deepcopy(layout)
    text_iter = iter(texts_other)
    for box in twin.boxes:
        if box.type == "text":
            try:
                box.payload = next(text_iter)[:200]
            except StopIteration:
                box.payload = ""
    return twin


def _to_sample(
    sid: str,
    image_path: str,
    page_size: tuple[int, int],
    annotations: list[dict[str, Any]],
    language: str,
    subset_name: str,
) -> TextAtlasSample:
    ocr_lines: list[OcrLine] = []
    rendered: list[str] = []
    for ann in annotations:
        if ann["type"] == "text":
            x0, y0, x1, y1 = ann["bbox"]
            ocr_lines.append(OcrLine(text=ann["text"], bbox=BBox.from_xyxy(x0, y0, x1, y1, label=ann.get("role"))))
            rendered.append(ann["text"])
    return TextAtlasSample(
        sample_id=sid,
        image_path=image_path,
        width=page_size[0], height=page_size[1],
        source_subset=subset_name,
        layout_type="interleaved",
        rendered_text="\n".join(rendered),
        scene_caption=("一张交错图文版式的中文页面" if language == "zh" else "An interleaved Chinese/English image-text page"),
        prompt="",
        language="zh-Hans" if language == "zh" else "en",
        ocr_lines=ocr_lines,
        metadata={"annotations": annotations},
    )


def build_text_vision_blend_parallel(
    sources: Iterable[dict[str, Any]],
    output_dir: str | Path,
    config_path: str | Path | None = None,
    seed: int = 20260428,
) -> None:
    cfg = load_config(config_path)
    fonts_root = resolve_path(cfg, "fonts_root")
    en_fonts_root = resolve_path(cfg, "fonts_en_root")
    img_zh = Path(output_dir) / "images_zh"
    img_en = Path(output_dir) / "images_en"
    img_zh.mkdir(parents=True, exist_ok=True)
    img_en.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    with ParallelJsonlWriter(output_dir, "text_vision_blend_parallel", shard_size=cfg["export"]["shard_size"]) as writer:
        for entry in tqdm(sources):
            zh_font, en_font = sample_font_pair(fonts_root, en_fonts_root, rng=rng)
            if entry["type"] == "wit":
                layout_zh = plan_wit(entry["image"], entry["sections_zh"])
                layout_en = plan_wit(entry["image"], entry["sections_en"])
            else:
                # Same image positions, parallel text payloads.
                layout_zh = plan_obelics(entry["images"], entry["texts_zh"], rng=random.Random(seed + 1))
                layout_en = _twin_layout(layout_zh, entry["texts_en"])

            try:
                png_zh, ann_zh = _render_layout(layout_zh, zh_font["path"], "zh")
                png_en, ann_en = _render_layout(layout_en, en_font["path"], "en")
            except Exception:
                continue
            sid = stable_id("blend", entry.get("id"), zh_font["name"])
            zh_path = save_image(Image.open(io.BytesIO(png_zh)), img_zh, sid, fmt="png")
            en_path = save_image(Image.open(io.BytesIO(png_en)), img_en, sid, fmt="png")

            zh_sample = _to_sample(f"{sid}-zh", zh_path, (int(layout_zh.page_w), int(layout_zh.page_h)), ann_zh, "zh", "TextVisionBlend-Parallel/zh")
            en_sample = _to_sample(f"{sid}-en", en_path, (int(layout_en.page_w), int(layout_en.page_h)), ann_en, "en", "TextVisionBlend-Parallel/en")
            zh_sample.prompt = "请生成一个交错图文页面，包含以下中文文字与对应图像。"
            en_sample.prompt = "Generate an interleaved page that contains the following English text blocks and matching images."

            writer.write(ParallelTextAtlasSample(
                pair_id=sid,
                parallelism="shared_layout",
                layout_type="interleaved",
                source_subset="TextVisionBlend-Parallel",
                zh=zh_sample, en=en_sample,
                shared={
                    "page_size": [layout_zh.page_w, layout_zh.page_h],
                    "image_boxes": [b.__dict__ for b in layout_zh.boxes if b.type == "image"],
                    "text_boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "role": b.role} for b in layout_zh.boxes if b.type == "text"],
                    "font_pair_style": zh_font.get("shared_style"),
                },
                alignment=AlignmentInfo(source=entry.get("id", ""), method="metadata_pair"),
                metadata={"font_zh": zh_font, "font_en": en_font, "entry_type": entry["type"]},
            ))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", default="data/output/text_vision_blend_parallel")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    sources = []
    with open(args.sources, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sources.append(json.loads(line))
    build_text_vision_blend_parallel(sources, args.output, args.config, args.seed)


if __name__ == "__main__":
    cli()
