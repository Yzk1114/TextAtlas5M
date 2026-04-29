"""Bilingual / parallel PPT2Details.

The slide image itself is identical across languages (same PPTX → same PNG).
Only the descriptive prompt differs:
- ``zh``: the Chinese version of the paper's PPT2Details prompt.
- ``en``: the original (English) PPT2Details prompt verbatim from the paper.

We additionally translate the OCR text via the configured translator so each
side has a self-contained ``rendered_text`` field.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import save_image, stable_id
from ...common.llm import LLMClient
from ...common.ocr import ChineseOCR
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.schema import OcrLine, TextAtlasSample
from ...common.translate import Translator, cross_lingual_similarity


PPT2DETAILS_PROMPT_ZH = (
    "给定一张中文幻灯片图像，请将其中所有可见的视觉元素（包括文字段落、图表、表格、流程图等）"
    "整合为一段连贯、流畅、逻辑一致的中文描述。\n要求：\n"
    "1. 完整保留所有文字内容（含数字、专有名词、标点）；\n2. 描述所有非文字视觉元素；\n"
    "3. 不得遗漏或改写关键短语；\n4. 仅输出一段话，不要分点。"
)
PPT2DETAILS_PROMPT_EN = (
    "Given a PowerPoint slide image, extract and summarize all visual elements—such as text blocks, "
    "charts, tables, and diagrams—into a single, fluent, and logically consistent paragraph. "
    "You must: 1. Accurately preserve all textual content and wording details. 2. Include descriptions of "
    "all visual elements (e.g., diagrams, tables) if present. 3. Avoid omitting or paraphrasing key phrases. "
    "4. Output only one paragraph per slide."
)


def _pptx_to_pdf(pptx: Path, work_dir: Path) -> Path | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(work_dir), str(pptx)],
        capture_output=True, timeout=300,
    )
    return (work_dir / (pptx.stem + ".pdf")) if res.returncode == 0 else None


def _pdf_to_images(pdf_path: Path, dpi: int = 150) -> Iterable[Image.Image]:
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()


def build_ppt2details_parallel(
    pptx_files: Iterable[Path],
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images_shared"
    work_dir = Path(output_dir) / "_work"
    image_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("soffice") is None:
        raise RuntimeError("LibreOffice (`soffice`) not found in PATH; required for PPT→PDF conversion.")

    vlm = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"], cache_dir=cache_root)
    ocr = ChineseOCR(primary=cfg["ocr"]["primary"])
    translator = Translator(provider=cfg["parallel"]["translate_provider"], model=cfg["parallel"]["translate_model"], cache_dir=cache_root)

    with ParallelJsonlWriter(output_dir, "ppt2details_parallel", shard_size=cfg["export"]["shard_size"]) as writer:
        for pptx in tqdm(pptx_files):
            pptx = Path(pptx)
            pdf = _pptx_to_pdf(pptx, work_dir)
            if pdf is None:
                continue
            for page_idx, img in enumerate(_pdf_to_images(pdf)):
                ocr_result = ocr.read(img)
                if ocr_result.char_count < 5:
                    continue
                sid = stable_id(pptx.name, page_idx)
                img_path = save_image(img, image_dir, sid, fmt="png")
                try:
                    desc_zh = vlm.chat(prompt=PPT2DETAILS_PROMPT_ZH, images=[img_path], temperature=0.4, max_tokens=2048).text.strip()
                    desc_en = vlm.chat(prompt=PPT2DETAILS_PROMPT_EN, images=[img_path], temperature=0.4, max_tokens=2048).text.strip()
                except Exception:
                    continue
                if not desc_zh or not desc_en:
                    continue
                ocr_zh = ocr_result.text
                try:
                    ocr_en = translator.translate(ocr_zh, direction="zh2en").text
                except Exception:
                    ocr_en = ""
                try:
                    sim = cross_lingual_similarity(desc_zh, desc_en)
                except Exception:
                    sim = None

                shared_meta = {"source_pptx": str(pptx), "page_idx": page_idx, "image_sha": sid}
                zh_sample = TextAtlasSample(
                    sample_id=f"{sid}-zh", image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="PPT2Details-Parallel/zh",
                    layout_type="ppt_details", language="zh-Hans",
                    rendered_text=ocr_zh, scene_caption="一张中文 PowerPoint 幻灯片", prompt=desc_zh,
                    ocr_lines=[OcrLine(text=l.text, bbox=l.bbox, confidence=l.confidence) for l in ocr_result.lines],
                    metadata=shared_meta,
                )
                en_sample = TextAtlasSample(
                    sample_id=f"{sid}-en", image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="PPT2Details-Parallel/en",
                    layout_type="ppt_details", language="en",
                    rendered_text=ocr_en, scene_caption="A PowerPoint slide image.", prompt=desc_en,
                    ocr_lines=[OcrLine(text=l.text, bbox=l.bbox, confidence=l.confidence) for l in ocr_result.lines],
                    metadata=shared_meta,
                )
                writer.write(ParallelTextAtlasSample(
                    pair_id=sid,
                    parallelism="same_image",
                    layout_type="ppt_details",
                    source_subset="PPT2Details-Parallel",
                    zh=zh_sample, en=en_sample,
                    shared={"image_path": img_path, **shared_meta},
                    alignment=AlignmentInfo(
                        source=str(pptx), method="vlm_dual",
                        forward_model=cfg["vlm"]["default_model"], bge_m3_sim=sim,
                        len_zh_chars=len(desc_zh), len_en_chars=len(desc_en),
                    ),
                ))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx-glob", required=True)
    parser.add_argument("--output", default="data/output/ppt2details_parallel")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    files = sorted(Path().glob(args.pptx_glob))
    build_ppt2details_parallel(files, args.output, args.config)


if __name__ == "__main__":
    cli()
