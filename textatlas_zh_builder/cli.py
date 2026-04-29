from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_nested, load_config
from .filtering import filter_long_words_jsonl
from .fonts import FontCatalog
from .interleave import InterleavedConfig, build_interleaved_dataset
from .pdf_extract import extract_pdf_pages
from .rendering import RenderConfig, render_many_clean_text
from .schema import write_jsonl
from .text_utils import TextFilterConfig, normalize_text


def _load_texts(path: str | Path) -> list[str]:
    input_path = Path(path)
    texts: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if input_path.suffix.lower() == ".jsonl":
                record = json.loads(line)
                text = record.get("text") or record.get("caption") or record.get("prompt")
            else:
                text = line
            if text:
                texts.append(normalize_text(str(text)))
    return texts


def build_clean(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    texts = _load_texts(args.input)
    font_dirs = args.font_dir or get_nested(config, "fonts", "directories", default=None)
    fonts = FontCatalog.from_directories(font_dirs)
    render_config = RenderConfig(
        width=args.width,
        height=args.height,
        margin=get_nested(config, "render", "margin", default=64),
        min_font_size=get_nested(config, "render", "min_font_size", default=24),
        max_font_size=get_nested(config, "render", "max_font_size", default=56),
        max_units=args.max_units,
    )
    samples = render_many_clean_text(texts[: args.limit], args.output_dir, fonts, render_config, seed=args.seed)
    count = write_jsonl(samples, args.output_jsonl)
    print(f"Wrote {count} CleanTextSynthZH samples to {args.output_jsonl}")


def build_interleave(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    font_dirs = args.font_dir or get_nested(config, "fonts", "directories", default=None)
    fonts = FontCatalog.from_directories(font_dirs)
    samples = build_interleaved_dataset(
        args.input,
        args.output_dir,
        font_catalog=fonts,
        config=InterleavedConfig(
            page_width=args.width,
            page_height=args.height,
            margin=get_nested(config, "interleave", "margin", default=48),
            min_font_size=get_nested(config, "interleave", "min_font_size", default=18),
            max_font_size=get_nested(config, "interleave", "max_font_size", default=30),
            max_text_units_per_box=get_nested(config, "interleave", "max_text_units_per_box", default=80),
        ),
        seed=args.seed,
        limit=args.limit,
    )
    count = write_jsonl(samples, args.output_jsonl)
    print(f"Wrote {count} TextVisionBlendZH samples to {args.output_jsonl}")


def build_filter_ocr(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    text_filter = TextFilterConfig(
        min_units=get_nested(config, "filter", "min_units", default=10),
        min_unique_ratio=get_nested(config, "filter", "min_unique_ratio", default=0.3),
        max_consecutive_repeat=get_nested(config, "filter", "max_consecutive_repeat", default=3),
        min_cjk_ratio=get_nested(config, "filter", "min_cjk_ratio", default=0.0),
    )
    count = filter_long_words_jsonl(
        args.input,
        args.output_jsonl,
        config=text_filter,
        subset=args.subset,
    )
    print(f"Wrote {count} filtered OCR samples to {args.output_jsonl}")


def build_pdf(args: argparse.Namespace) -> None:
    samples = []
    for pdf_path in sorted(Path(args.input_dir).glob("*.pdf"))[: args.limit]:
        samples.extend(extract_pdf_pages(pdf_path, args.output_dir, subset=args.subset))
    count = write_jsonl(samples, args.output_jsonl)
    print(f"Wrote {count} PDF-derived samples to {args.output_jsonl}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="textatlas-zh",
        description="Reproduce TextAtlas-style Chinese dataset construction pipelines.",
    )
    subparsers = parser.add_subparsers(required=True)

    clean = subparsers.add_parser("clean-text", help="Render Chinese CleanTextSynth-style text-only images.")
    clean.add_argument("--input", required=True, help="TXT or JSONL file containing source Chinese text.")
    clean.add_argument("--output-dir", required=True, help="Directory for rendered PNG images.")
    clean.add_argument("--output-jsonl", required=True, help="Unified annotation JSONL output path.")
    clean.add_argument("--config", help="Optional YAML config.")
    clean.add_argument("--font-dir", action="append", help="Directory containing Chinese TTF/OTF fonts.")
    clean.add_argument("--limit", type=int, default=None, help="Maximum number of samples.")
    clean.add_argument("--seed", type=int, default=0)
    clean.add_argument("--width", type=int, default=1024)
    clean.add_argument("--height", type=int, default=1024)
    clean.add_argument("--max-units", type=int, default=None, help="Truncate by Chinese chars / Latin tokens.")
    clean.set_defaults(func=build_clean)

    interleave = subparsers.add_parser("interleave", help="Generate parseable interleaved PDF/PNG samples.")
    interleave.add_argument("--input", required=True, help="JSONL with text_segments/texts and optional images.")
    interleave.add_argument("--output-dir", required=True, help="Directory for generated PDFs and PNGs.")
    interleave.add_argument("--output-jsonl", required=True)
    interleave.add_argument("--config", help="Optional YAML config.")
    interleave.add_argument("--font-dir", action="append")
    interleave.add_argument("--limit", type=int, default=None)
    interleave.add_argument("--seed", type=int, default=0)
    interleave.add_argument("--width", type=int, default=1024)
    interleave.add_argument("--height", type=int, default=1024)
    interleave.set_defaults(func=build_interleave)

    ocr = subparsers.add_parser("filter-ocr", help="Filter OCR/caption JSONL into Chinese long-text samples.")
    ocr.add_argument("--input", required=True, help="Input JSONL with image_path and OCR text blocks.")
    ocr.add_argument("--output-jsonl", required=True)
    ocr.add_argument("--config", help="Optional YAML config.")
    ocr.add_argument("--image-root", help="Prefix for relative image paths.")
    ocr.add_argument("--min-text-blocks", type=int, default=1)
    ocr.add_argument("--subset", default="LongWordsSubsetZH")
    ocr.add_argument("--source")
    ocr.set_defaults(func=build_filter_ocr)

    pdf = subparsers.add_parser("pdf", help="Extract Paper2Text-style Chinese samples from PDFs.")
    pdf.add_argument("--input-dir", required=True, help="Directory containing PDF files.")
    pdf.add_argument("--output-dir", required=True, help="Directory for rendered PDF page images.")
    pdf.add_argument("--output-jsonl", required=True)
    pdf.add_argument("--subset", default="Paper2TextZH")
    pdf.add_argument("--limit", type=int, default=None)
    pdf.set_defaults(func=build_pdf)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
