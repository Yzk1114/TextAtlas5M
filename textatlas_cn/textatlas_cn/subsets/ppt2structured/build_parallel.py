"""Bilingual / parallel PPT2Structured.

Same image, parallel structured prompts. Element-level bbox/font/text are
shared; only the per-image element captions and the joined "structured"
prompt are produced in two languages.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import save_image, stable_id
from ...common.llm import LLMClient
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.schema import BBox, OcrLine, TextAtlasSample
from ...common.translate import Translator


CAPTION_PROMPT_ZH = "请用一段不超过80字的中文，描述这张图片的核心内容。"
CAPTION_PROMPT_EN = "Describe the core content of this image in no more than 80 English words."


def _extract_page(page: fitz.Page) -> tuple[Image.Image, list[dict]]:
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    elements = []
    for blk in page.get_text("dict")["blocks"]:
        if blk["type"] == 0:
            for line in blk["lines"]:
                for span in line["spans"]:
                    elements.append({"type": "text", "bbox": span["bbox"], "text": span["text"], "font": span["font"], "size": span["size"], "color": span["color"]})
        elif blk["type"] == 1:
            elements.append({"type": "image", "bbox": blk["bbox"], "image_bytes": blk.get("image"), "ext": blk.get("ext")})
    return img, elements


def build_ppt2structured_parallel(
    pdf_files: Iterable[Path],
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images_shared"
    crop_dir = Path(output_dir) / "crops"
    image_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    vlm = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"], cache_dir=cache_root)
    translator = Translator(provider=cfg["parallel"]["translate_provider"], model=cfg["parallel"]["translate_model"], cache_dir=cache_root)

    with ParallelJsonlWriter(output_dir, "ppt2structured_parallel", shard_size=cfg["export"]["shard_size"]) as writer:
        for pdf_path in tqdm(pdf_files):
            doc = fitz.open(pdf_path)
            for page_idx, page in enumerate(doc):
                img, elements = _extract_page(page)
                if not elements:
                    continue
                sid = stable_id(pdf_path.name, page_idx)
                img_path = save_image(img, image_dir, sid, fmt="png")

                ocr_lines: list[OcrLine] = []
                rendered: list[str] = []
                bullets_zh, bullets_en = [], []
                for i, el in enumerate(elements):
                    x0, y0, x1, y1 = el["bbox"]
                    bbox = BBox.from_xyxy(x0, y0, x1, y1, label=el["type"])
                    if el["type"] == "text":
                        ocr_lines.append(OcrLine(text=el["text"], bbox=bbox))
                        rendered.append(el["text"])
                        en_text = ""
                        try:
                            en_text = translator.translate(el["text"], direction="zh2en").text
                        except Exception:
                            en_text = el["text"]
                        bullets_zh.append(f"- 文字（{el['font']} {el['size']:.1f}pt @{[round(v,1) for v in el['bbox']]}）: {el['text']}")
                        bullets_en.append(f"- text ({el['font']} {el['size']:.1f}pt @{[round(v,1) for v in el['bbox']]}): {en_text}")
                    else:
                        crop_path = None
                        if el.get("image_bytes"):
                            crop = Image.open(io.BytesIO(el["image_bytes"]))
                            crop_path = save_image(crop, crop_dir, f"{sid}-img{i}", fmt=el.get("ext") or "png")
                        cap_zh = cap_en = ""
                        if crop_path:
                            try:
                                cap_zh = vlm.chat(CAPTION_PROMPT_ZH, images=[crop_path], temperature=0.4, max_tokens=256).text.strip()
                                cap_en = vlm.chat(CAPTION_PROMPT_EN, images=[crop_path], temperature=0.4, max_tokens=256).text.strip()
                            except Exception:
                                pass
                        bullets_zh.append(f"- 图片（@{[round(v,1) for v in el['bbox']]}）: {cap_zh}")
                        bullets_en.append(f"- image (@{[round(v,1) for v in el['bbox']]}): {cap_en}")

                shared = {"source_pdf": str(pdf_path), "page_idx": page_idx, "elements": elements}
                zh_sample = TextAtlasSample(
                    sample_id=f"{sid}-zh", image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="PPT2Structured-Parallel/zh",
                    layout_type="ppt_structured", language="zh-Hans",
                    rendered_text="\n".join(rendered), scene_caption="一页中文学术幻灯",
                    prompt="请按照以下结构化要素生成一页幻灯片：\n" + "\n".join(bullets_zh),
                    ocr_lines=ocr_lines, metadata=shared,
                )
                en_sample = TextAtlasSample(
                    sample_id=f"{sid}-en", image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="PPT2Structured-Parallel/en",
                    layout_type="ppt_structured", language="en",
                    rendered_text="\n".join(rendered), scene_caption="An academic presentation slide.",
                    prompt="Generate one slide that matches the following structured elements:\n" + "\n".join(bullets_en),
                    ocr_lines=ocr_lines, metadata=shared,
                )
                writer.write(ParallelTextAtlasSample(
                    pair_id=sid, parallelism="same_image", layout_type="ppt_structured",
                    source_subset="PPT2Structured-Parallel",
                    zh=zh_sample, en=en_sample,
                    shared={"image_path": img_path, **shared},
                    alignment=AlignmentInfo(source=str(pdf_path), method="vlm_dual+translate"),
                ))
            doc.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-glob", required=True)
    parser.add_argument("--output", default="data/output/ppt2structured_parallel")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    files = sorted(Path().glob(args.pdf_glob))
    build_ppt2structured_parallel(files, args.output, args.config)


if __name__ == "__main__":
    cli()
