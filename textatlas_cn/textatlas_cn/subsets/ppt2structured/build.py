"""Build PPT2Structured-CN.

For each Chinese AutoSlideGen-equivalent PDF (e.g. 机器之心 talks, ChinaXiv 配套幻灯)
we use PyMuPDF to extract every text/image bbox and font properties, then ask
Qwen2.5-VL to caption each embedded image. Output is a JSON of element-level
annotations preserved alongside the rendered slide PNG.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.llm import LLMClient
from ...common.schema import BBox, OcrLine, TextAtlasSample


IMAGE_CAPTION_PROMPT = "请用一段不超过80字的中文，描述这张图片的核心内容。"


def _extract_page(page: fitz.Page) -> tuple[Image.Image, list[dict[str, Any]]]:
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    elements: list[dict[str, Any]] = []
    blocks = page.get_text("dict")["blocks"]
    for blk in blocks:
        if blk["type"] == 0:  # text block
            for line in blk["lines"]:
                for span in line["spans"]:
                    elements.append({
                        "type": "text",
                        "bbox": span["bbox"],
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "color": span["color"],
                    })
        elif blk["type"] == 1:  # image block
            elements.append({
                "type": "image",
                "bbox": blk["bbox"],
                "image_bytes": blk.get("image"),
                "ext": blk.get("ext"),
            })
    return img, elements


def build_ppt2structured_cn(
    pdf_files: Iterable[Path],
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images"
    crop_dir = Path(output_dir) / "crops"
    meta_dir = Path(output_dir) / "metadata"
    for d in (image_dir, crop_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    vlm = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"], cache_dir=cache_root)

    with JsonlShardWriter(meta_dir, "ppt2structured_cn", shard_size=cfg["export"]["shard_size"]) as writer:
        for pdf_path in tqdm(pdf_files):
            doc = fitz.open(pdf_path)
            for page_idx, page in enumerate(doc):
                img, elements = _extract_page(page)
                if not elements:
                    continue
                sid = stable_id(pdf_path.name, page_idx)
                img_path = save_image(img, image_dir, sid, fmt="png")

                ocr_lines: list[OcrLine] = []
                rendered_lines: list[str] = []
                bullets: list[str] = []
                for i, el in enumerate(elements):
                    x0, y0, x1, y1 = el["bbox"]
                    bbox = BBox.from_xyxy(x0, y0, x1, y1, label=el["type"])
                    if el["type"] == "text":
                        ocr_lines.append(OcrLine(text=el["text"], bbox=bbox))
                        rendered_lines.append(el["text"])
                        bullets.append(
                            f"- 文字（{el['font']} {el['size']:.1f}pt @{[round(v,1) for v in el['bbox']]}）: {el['text']}"
                        )
                    else:
                        crop_path = None
                        if el.get("image_bytes"):
                            crop = Image.open(io.BytesIO(el["image_bytes"]))
                            crop_path = save_image(crop, crop_dir, f"{sid}-img{i}", fmt=el.get("ext") or "png")
                        cap = ""
                        if crop_path:
                            try:
                                cap = vlm.chat(IMAGE_CAPTION_PROMPT, images=[crop_path], temperature=0.4, max_tokens=256).text.strip()
                            except Exception:
                                cap = ""
                        el["caption"] = cap
                        bullets.append(f"- 图片（@{[round(v,1) for v in el['bbox']]}）: {cap}")

                sample = TextAtlasSample(
                    sample_id=sid,
                    image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="PPT2Structured-CN",
                    layout_type="ppt_structured",
                    rendered_text="\n".join(rendered_lines),
                    scene_caption="一页中文学术幻灯",
                    prompt="请按照以下结构化要素生成一页中文幻灯片：\n" + "\n".join(bullets),
                    ocr_lines=ocr_lines,
                    metadata={
                        "source_pdf": str(pdf_path),
                        "page_idx": page_idx,
                        "elements": elements,
                    },
                )
                writer.write(sample.to_dict())
            doc.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-glob", required=True)
    parser.add_argument("--output", default="data/output/ppt2structured_cn")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    files = sorted(Path().glob(args.pdf_glob))
    build_ppt2structured_cn(files, args.output, args.config)


if __name__ == "__main__":
    cli()
