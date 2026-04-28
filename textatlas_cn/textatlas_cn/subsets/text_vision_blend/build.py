"""Build TextVisionBlend-CN.

Pipeline:
1. Sample a Chinese-Wiki / WIT-zh entry (preferred) or fall back to Obelics-CN.
2. Choose layout planner (WIT vs. Obelics).
3. Use PyMuPDF to render a parseable PDF (Chinese fonts embedded).
4. Save the rendered page as a PNG, plus a JSON describing every text/image bbox.
5. Generate Chinese caption for each image via VLM (Qwen2.5-VL by default) – ≤50 chars.
6. Aggregate into the unified TextAtlasSample schema.
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import requests
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.fonts import sample_font
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.llm import LLMClient
from ...common.schema import BBox, FontAttrs, OcrLine, TextAtlasSample
from .layout import PageLayout, plan_obelics, plan_wit


CAPTION_PROMPT = "请用不超过50字的中文为这张图像生成简洁的描述，不要照抄图中文字。"


def _render_layout_to_pdf(layout: PageLayout, font_path: str) -> tuple[bytes, list[dict[str, Any]]]:
    """Render the layout via PyMuPDF and return (PNG bytes, annotation list)."""
    doc = fitz.open()
    page = doc.new_page(width=layout.page_w, height=layout.page_h)
    page.insert_font(fontname="ZH", fontfile=font_path)
    annotations: list[dict[str, Any]] = []

    for box in layout.boxes:
        rect = fitz.Rect(box.x, box.y, box.x + box.w, box.y + box.h)
        if box.type == "text" and box.payload:
            html = (
                f'<div style="font-family:ZH;font-size:14pt;line-height:1.4;'
                f'color:#202020;text-align:justify;">{box.payload}</div>'
            )
            page.insert_htmlbox(rect, html)
            annotations.append({
                "type": "text", "role": box.role, "bbox": [box.x, box.y, box.x + box.w, box.y + box.h],
                "text": box.payload, "font": "ZH",
            })
        elif box.type == "image" and box.payload:
            try:
                img_bytes = requests.get(box.payload, timeout=20).content if box.payload.startswith("http") else Path(box.payload).read_bytes()
                page.insert_image(rect, stream=img_bytes, keep_proportion=True)
                annotations.append({
                    "type": "image", "role": box.role, "bbox": [box.x, box.y, box.x + box.w, box.y + box.h],
                    "src": box.payload,
                })
            except Exception:
                continue

    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes, annotations


def _generate_image_captions(annotations: list[dict[str, Any]], llm: LLMClient) -> dict[str, str]:
    """Call Qwen-VL on each embedded image and store captions keyed by src."""
    captions: dict[str, str] = {}
    for ann in annotations:
        if ann["type"] != "image":
            continue
        src = ann.get("src")
        if not src or src in captions:
            continue
        try:
            resp = llm.chat(
                prompt=CAPTION_PROMPT, images=[src], temperature=0.4, max_tokens=128,
            )
            captions[src] = resp.text.strip()
        except Exception:
            captions[src] = ""
    return captions


def _annotations_to_sample(
    sample_id: str, image_path: str, page_size: tuple[int, int],
    annotations: list[dict[str, Any]], image_captions: dict[str, str],
) -> TextAtlasSample:
    ocr_lines: list[OcrLine] = []
    rendered_lines: list[str] = []
    for ann in annotations:
        if ann["type"] == "text":
            x0, y0, x1, y1 = ann["bbox"]
            ocr_lines.append(OcrLine(text=ann["text"], bbox=BBox.from_xyxy(x0, y0, x1, y1, label=ann.get("role"))))
            rendered_lines.append(ann["text"])
        elif ann["type"] == "image":
            ann["caption"] = image_captions.get(ann.get("src"), "")
    rendered_text = "\n".join(rendered_lines)

    bullets = []
    for ann in annotations:
        if ann["type"] == "text":
            bullets.append(f"- 文字（{ann.get('role') or 'block'}@{ann['bbox']}）: {ann['text']}")
        else:
            cap = ann.get("caption") or "（图像，无字幕）"
            bullets.append(f"- 图片（@{ann['bbox']}）: {cap}")
    prompt = "请生成一个交错图文页面，包含以下元素：\n" + "\n".join(bullets)

    return TextAtlasSample(
        sample_id=sample_id,
        image_path=image_path,
        width=page_size[0], height=page_size[1],
        source_subset="TextVisionBlend-CN",
        layout_type="interleaved",
        rendered_text=rendered_text,
        scene_caption="一张交错图文版式的中文页面",
        prompt=prompt,
        ocr_lines=ocr_lines,
        metadata={"annotations": annotations},
    )


def build_text_vision_blend_cn(
    sources: Iterable[dict[str, Any]],
    output_dir: str | Path,
    config_path: str | Path | None = None,
    seed: int = 20260428,
) -> None:
    cfg = load_config(config_path)
    fonts_root = resolve_path(cfg, "fonts_root")
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    llm = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"],
                    cache_dir=resolve_path(cfg, "cache_root"))

    with JsonlShardWriter(meta_dir, "text_vision_blend_cn", shard_size=cfg["export"]["shard_size"]) as writer:
        for entry in tqdm(sources):
            font = sample_font(fonts_root, family="宋体" if rng.random() < 0.6 else None, rng=rng)
            if entry["type"] == "wit":
                layout = plan_wit(entry["image"], entry["sections"])
            else:
                layout = plan_obelics(entry["images"], entry["texts"], rng=rng)

            try:
                png_bytes, annotations = _render_layout_to_pdf(layout, font["path"])
            except Exception:
                continue
            sid = stable_id(entry.get("id"), font["name"], rng.random())
            img = Image.open(io.BytesIO(png_bytes))
            img_path = save_image(img, image_dir, sid, fmt="png")

            captions = _generate_image_captions(annotations, llm)
            sample = _annotations_to_sample(sid, img_path, img.size, annotations, captions)
            sample.metadata["font"] = font
            sample.metadata["source"] = entry["type"]
            writer.write(sample.to_dict())


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="Path to JSONL with prepared {type,images,texts,sections} entries")
    parser.add_argument("--output", default="data/output/text_vision_blend_cn")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    sources = []
    with open(args.sources, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sources.append(json.loads(line))
    build_text_vision_blend_cn(sources, args.output, args.config)


if __name__ == "__main__":
    cli()
