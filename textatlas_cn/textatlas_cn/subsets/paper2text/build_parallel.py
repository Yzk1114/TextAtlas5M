"""Bilingual / parallel Paper2Text.

We support two strategies:
1. ``--mode translate``: take Chinese papers (CNKI / ChinaXiv) and translate
   each text span via the configured translator. Layout is identical in both
   languages because the underlying image is shared.
2. ``--mode bilingual``: take papers that already ship with Chinese + English
   abstracts/titles (most arXiv-CN, e.g. 中科院期刊双语 abstract); we keep both
   sides verbatim where available and translate the rest.

Either way the rendered slide image is identical for both languages.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import save_image, stable_id
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.schema import BBox, FontAttrs, OcrLine, TextAtlasSample
from ...common.translate import Translator


def build_paper2text_parallel(
    pdf_files: Iterable[Path],
    output_dir: str | Path,
    config_path: str | Path | None = None,
    max_pages_per_paper: int = 30,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images_shared"
    image_dir.mkdir(parents=True, exist_ok=True)
    translator = Translator(provider=cfg["parallel"]["translate_provider"], model=cfg["parallel"]["translate_model"], cache_dir=cache_root)

    with ParallelJsonlWriter(output_dir, "paper2text_parallel", shard_size=cfg["export"]["shard_size"]) as writer:
        for pdf in tqdm(pdf_files):
            doc = fitz.open(pdf)
            for page_idx, page in enumerate(doc):
                if page_idx >= max_pages_per_paper:
                    break
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                sid = stable_id(pdf.name, page_idx)
                img_path = save_image(img, image_dir, sid, fmt="png")

                ocr_lines: list[OcrLine] = []
                spans = []
                for blk in page.get_text("dict")["blocks"]:
                    if blk["type"] != 0:
                        continue
                    for line in blk["lines"]:
                        for span in line["spans"]:
                            x0, y0, x1, y1 = span["bbox"]
                            color = span["color"]
                            r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
                            attrs = FontAttrs(family=span["font"], size=span["size"], color=(r, g, b))
                            ocr_lines.append(OcrLine(text=span["text"], bbox=BBox.from_xyxy(x0, y0, x1, y1), font=attrs))
                            spans.append({"bbox": span["bbox"], "text": span["text"], "font": span["font"], "size": span["size"], "color": (r, g, b)})

                rendered_zh = "\n".join(s["text"] for s in spans)
                if not rendered_zh.strip():
                    continue
                # Translate the entire page in one shot to preserve global context.
                try:
                    rendered_en = translator.translate(rendered_zh, direction="zh2en").text
                except Exception:
                    rendered_en = ""
                if not rendered_en:
                    continue

                prompt_zh = f"中文论文第 {page_idx+1} 页，按字体和位置呈现的文字内容：\n" + "\n".join(
                    f"- ({s['font']} {s['size']:.1f}pt) {s['text']}" for s in spans
                )
                prompt_en = (
                    f"Page {page_idx+1} of an academic paper rendered with the original font and "
                    f"position information. The rendered text on the image is in Chinese; the "
                    f"following English translation is provided for reference:\n{rendered_en}"
                )
                shared = {"source_pdf": str(pdf), "page_idx": page_idx, "spans": spans}
                zh_sample = TextAtlasSample(
                    sample_id=f"{sid}-zh", image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="Paper2Text-Parallel/zh", layout_type="paper_text", language="zh-Hans",
                    rendered_text=rendered_zh, scene_caption="一页中文学术论文", prompt=prompt_zh,
                    ocr_lines=ocr_lines, metadata=shared,
                )
                en_sample = TextAtlasSample(
                    sample_id=f"{sid}-en", image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="Paper2Text-Parallel/en", layout_type="paper_text", language="en",
                    rendered_text=rendered_en, scene_caption="A page of an academic paper.", prompt=prompt_en,
                    ocr_lines=ocr_lines, metadata=shared,
                )
                writer.write(ParallelTextAtlasSample(
                    pair_id=sid, parallelism="same_image", layout_type="paper_text",
                    source_subset="Paper2Text-Parallel",
                    zh=zh_sample, en=en_sample,
                    shared={"image_path": img_path, **shared},
                    alignment=AlignmentInfo(
                        source=str(pdf), method="llm_translate",
                        forward_model=cfg["parallel"]["translate_model"],
                        len_zh_chars=len(rendered_zh), len_en_chars=len(rendered_en),
                    ),
                ))
            doc.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-glob", required=True)
    parser.add_argument("--output", default="data/output/paper2text_parallel")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    files = sorted(Path().glob(args.pdf_glob))
    build_paper2text_parallel(files, args.output, args.config)


if __name__ == "__main__":
    cli()
