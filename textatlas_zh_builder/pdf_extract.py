from __future__ import annotations

from pathlib import Path

import fitz

from .schema import DatasetSample, ImageBlock, TextBlock
from .text_utils import normalize_text, stable_id


def _rgb_from_int(color: int) -> tuple[int, int, int]:
    return ((color >> 16) & 255, (color >> 8) & 255, color & 255)


def extract_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    subset: str = "Paper2TextZH",
    render_scale: float = 2.0,
) -> list[DatasetSample]:
    """Extract rendered pages and structured text/image blocks from a PDF.

    This reproduces the Paper2Text/PPT2Structured branch of the paper: render
    pages as images, preserve text content, bbox, font family, font size, and
    coarse image block positions through PyMuPDF.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    samples: list[DatasetSample] = []
    matrix = fitz.Matrix(render_scale, render_scale)
    for page_index, page in enumerate(document):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = image_dir / f"{pdf_path.stem}_page_{page_index + 1:04d}.png"
        pix.save(image_path)

        text_blocks: list[TextBlock] = []
        order = 0
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = normalize_text(span.get("text", ""))
                    if not text:
                        continue
                    bbox = tuple(float(v) * render_scale for v in span["bbox"])
                    text_blocks.append(
                        TextBlock(
                            text=text,
                            bbox=bbox,  # type: ignore[arg-type]
                            font=span.get("font"),
                            font_size=float(span.get("size", 0)) * render_scale,
                            color=_rgb_from_int(int(span.get("color", 0))),
                            reading_order=order,
                        )
                    )
                    order += 1

        image_blocks: list[ImageBlock] = []
        for image_order, info in enumerate(page.get_image_info(xrefs=True)):
            bbox = tuple(float(v) * render_scale for v in info["bbox"])
            image_blocks.append(
                ImageBlock(path="", bbox=bbox, caption=None, reading_order=image_order)  # type: ignore[arg-type]
            )

        text = "".join(block.text for block in sorted(text_blocks, key=lambda item: (item.bbox[1], item.bbox[0])))
        prompt = f"生成一页中文文档图片，保持版面结构、字体大小和文字位置。页面文字包括：{text}"
        samples.append(
            DatasetSample(
                sample_id=stable_id(str(pdf_path), str(page_index), prefix="pdf_zh_"),
                subset=subset,
                image_path=str(image_path),
                prompt=prompt,
                source=str(pdf_path),
                text_blocks=text_blocks,
                image_blocks=image_blocks,
                metadata={"page_index": page_index, "render_scale": render_scale},
            )
        )
    document.close()
    return samples
