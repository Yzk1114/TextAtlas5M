"""Build PPT2Details-CN.

Pipeline:
1. PPTX → PDF via headless LibreOffice (`soffice --convert-to pdf`).
2. PDF → page PNGs via PyMuPDF.
3. Filter out slides with very little text (OCR <= 5 chars) and decorative title pages.
4. Call Qwen2.5-VL with the *Chinese* analogue of the paper's PPT2Details prompt
   to produce a single fluent paragraph per slide.
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
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.llm import LLMClient
from ...common.ocr import ChineseOCR
from ...common.schema import OcrLine, TextAtlasSample


PPT2DETAILS_PROMPT_CN = (
    "给定一张中文幻灯片图像，请将其中所有可见的视觉元素（包括文字段落、图表、表格、流程图等）"
    "整合为一段连贯、流畅、逻辑一致的中文描述。\n"
    "要求：\n"
    "1. 完整保留所有文字内容（含数字、专有名词、标点）；\n"
    "2. 描述所有非文字视觉元素；\n"
    "3. 不得遗漏或改写关键短语；\n"
    "4. 仅输出一段话，不要分点。"
)


def _pptx_to_pdf(pptx: Path, work_dir: Path) -> Path | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(work_dir), str(pptx)],
        capture_output=True, timeout=300,
    )
    if res.returncode != 0:
        return None
    out = work_dir / (pptx.stem + ".pdf")
    return out if out.exists() else None


def _pdf_to_images(pdf_path: Path, dpi: int = 150) -> Iterable[Image.Image]:
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()


def build_ppt2details_cn(
    pptx_files: Iterable[Path],
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    work_dir = Path(output_dir) / "_work"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("soffice") is None:
        raise RuntimeError("LibreOffice (`soffice`) not found in PATH; required for PPT→PDF conversion.")

    vlm = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"], cache_dir=cache_root)
    ocr = ChineseOCR(primary=cfg["ocr"]["primary"])

    with JsonlShardWriter(meta_dir, "ppt2details_cn", shard_size=cfg["export"]["shard_size"]) as writer:
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
                    resp = vlm.chat(prompt=PPT2DETAILS_PROMPT_CN, images=[img_path], temperature=0.4, max_tokens=2048)
                    description = resp.text.strip()
                except Exception:
                    description = ""
                if not description:
                    continue
                sample = TextAtlasSample(
                    sample_id=sid,
                    image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="PPT2Details-CN",
                    layout_type="ppt_details",
                    rendered_text=ocr_result.text,
                    scene_caption="一张中文 PowerPoint 幻灯片",
                    prompt=description,
                    ocr_lines=[OcrLine(text=l.text, bbox=l.bbox, confidence=l.confidence) for l in ocr_result.lines],
                    metadata={"source_pptx": str(pptx), "page_idx": page_idx},
                )
                writer.write(sample.to_dict())


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx-glob", required=True, help="Glob pattern for input pptx files")
    parser.add_argument("--output", default="data/output/ppt2details_cn")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    files = sorted(Path().glob(args.pptx_glob))
    build_ppt2details_cn(files, args.output, args.config)


if __name__ == "__main__":
    cli()
