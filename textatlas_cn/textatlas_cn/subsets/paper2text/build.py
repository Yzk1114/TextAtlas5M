"""Build Paper2Text-CN.

Per Chinese paper (CNKI 开放 / ChinaXiv / journals):
- For every page, render PNG (200dpi).
- Extract every text span: bbox, font name, size, color (PyMuPDF).
- Aggregate spans into a single textual prompt that lists font + position attributes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.schema import BBox, FontAttrs, OcrLine, TextAtlasSample


def build_paper2text_cn(
    pdf_files: Iterable[Path],
    output_dir: str | Path,
    config_path: str | Path | None = None,
    max_pages_per_paper: int = 30,
) -> None:
    cfg = load_config(config_path)
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with JsonlShardWriter(meta_dir, "paper2text_cn", shard_size=cfg["export"]["shard_size"]) as writer:
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
                spans: list[dict[str, Any]] = []
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

                rendered_text = "\n".join(s["text"] for s in spans)
                prompt = "中文论文第 {idx} 页，按字体和位置呈现的文字内容：\n".format(idx=page_idx + 1) + "\n".join(
                    f"- ({s['font']} {s['size']:.1f}pt) {s['text']}" for s in spans
                )
                sample = TextAtlasSample(
                    sample_id=sid,
                    image_path=img_path,
                    width=img.size[0], height=img.size[1],
                    source_subset="Paper2Text-CN",
                    layout_type="paper_text",
                    rendered_text=rendered_text,
                    scene_caption="一页中文学术论文",
                    prompt=prompt,
                    ocr_lines=ocr_lines,
                    metadata={"source_pdf": str(pdf), "page_idx": page_idx, "spans": spans},
                )
                writer.write(sample.to_dict())
            doc.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-glob", required=True)
    parser.add_argument("--output", default="data/output/paper2text_cn")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    files = sorted(Path().glob(args.pdf_glob))
    build_paper2text_cn(files, args.output, args.config)


if __name__ == "__main__":
    cli()
