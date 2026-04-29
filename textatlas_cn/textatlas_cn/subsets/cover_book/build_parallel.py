"""Bilingual / parallel CoverBook.

Two strategies:
1. **Edition pair**: when an input record contains both a Chinese edition and
   an English edition of the same work (matched by ISBN-13 → OpenLibrary), we
   keep both cover images and both metadata captions.  ``parallelism =
   "shared_layout"`` because the covers are different but reference the same
   book.
2. **Translate-only**: when only one cover is available, we keep the same
   image for both languages and translate the metadata caption.

Input JSONL fields per row:
    {"isbn", "title_zh", "title_en", "author_zh", "author_en",
     "publisher_zh", "publisher_en", "year", "category_zh", "category_en",
     "cover_url_zh", "cover_url_en"}
Missing fields are filled by the translator.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import save_image, stable_id
from ...common.llm import LLMClient
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.schema import OcrLine, TextAtlasSample
from ...common.translate import Translator


def _make_prompt(meta: dict, lang: str) -> str:
    if lang == "zh":
        parts = [meta.get("title_zh", "")]
        for k, p in [("author_zh", "作者"), ("publisher_zh", "出版社"), ("year", "出版年"), ("category_zh", "类别")]:
            if meta.get(k):
                parts.append(f"{p}：{meta[k]}")
        return "一本中文图书的封面，包含以下信息：" + " | ".join(p for p in parts if p)
    parts = [meta.get("title_en", "")]
    for k, p in [("author_en", "Author"), ("publisher_en", "Publisher"), ("year", "Year"), ("category_en", "Category")]:
        if meta.get(k):
            parts.append(f"{p}: {meta[k]}")
    return "Cover of an English-edition book with the following metadata: " + " | ".join(p for p in parts if p)


def _ensure_bilingual(meta: dict, translator: Translator) -> dict:
    out = dict(meta)
    if not out.get("title_en") and out.get("title_zh"):
        out["title_en"] = translator.translate(out["title_zh"], "zh2en").text
    if not out.get("title_zh") and out.get("title_en"):
        out["title_zh"] = translator.translate(out["title_en"], "en2zh").text
    if not out.get("author_en") and out.get("author_zh"):
        out["author_en"] = translator.translate(out["author_zh"], "zh2en").text
    if not out.get("publisher_en") and out.get("publisher_zh"):
        out["publisher_en"] = translator.translate(out["publisher_zh"], "zh2en").text
    if not out.get("category_en") and out.get("category_zh"):
        out["category_en"] = translator.translate(out["category_zh"], "zh2en").text
    return out


def build_cover_book_parallel(
    metadata_jsonl: Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    img_zh = Path(output_dir) / "images_zh"
    img_en = Path(output_dir) / "images_en"
    img_zh.mkdir(parents=True, exist_ok=True)
    img_en.mkdir(parents=True, exist_ok=True)
    cache_root = resolve_path(cfg, "cache_root")
    translator = Translator(provider=cfg["parallel"]["translate_provider"], model=cfg["parallel"]["translate_model"], cache_dir=cache_root)

    with ParallelJsonlWriter(output_dir, "cover_book_parallel", shard_size=cfg["export"]["shard_size"]) as writer, \
            metadata_jsonl.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh):
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            meta = _ensure_bilingual(meta, translator)
            url_zh = meta.get("cover_url_zh") or meta.get("cover_url")
            url_en = meta.get("cover_url_en") or url_zh
            try:
                img_zh_obj = Image.open(io.BytesIO(requests.get(url_zh, timeout=20).content)).convert("RGB")
                img_en_obj = Image.open(io.BytesIO(requests.get(url_en, timeout=20).content)).convert("RGB") if url_en != url_zh else img_zh_obj
            except Exception:
                continue

            sid = stable_id(meta.get("isbn") or meta.get("title_zh"), meta.get("title_en"))
            zh_path = save_image(img_zh_obj, img_zh, sid, fmt="jpg", quality=92)
            en_path = save_image(img_en_obj, img_en, sid, fmt="jpg", quality=92)

            zh_rendered = " ".join(filter(None, [meta.get("title_zh"), meta.get("author_zh"), meta.get("publisher_zh")]))
            en_rendered = " ".join(filter(None, [meta.get("title_en"), meta.get("author_en"), meta.get("publisher_en")]))
            zh_sample = TextAtlasSample(
                sample_id=f"{sid}-zh", image_path=zh_path,
                width=img_zh_obj.size[0], height=img_zh_obj.size[1],
                source_subset="CoverBook-Parallel/zh", layout_type="cover_book", language="zh-Hans",
                rendered_text=zh_rendered, scene_caption="中文图书封面", prompt=_make_prompt(meta, "zh"),
                ocr_lines=[OcrLine(text=zh_rendered, bbox=None)] if zh_rendered else [],
                metadata={"book": meta},
            )
            en_sample = TextAtlasSample(
                sample_id=f"{sid}-en", image_path=en_path,
                width=img_en_obj.size[0], height=img_en_obj.size[1],
                source_subset="CoverBook-Parallel/en", layout_type="cover_book", language="en",
                rendered_text=en_rendered, scene_caption="English-edition book cover", prompt=_make_prompt(meta, "en"),
                ocr_lines=[OcrLine(text=en_rendered, bbox=None)] if en_rendered else [],
                metadata={"book": meta},
            )
            mode = "shared_layout" if (url_en and url_en != url_zh) else "same_image"
            writer.write(ParallelTextAtlasSample(
                pair_id=sid,
                parallelism=mode,
                layout_type="cover_book",
                source_subset="CoverBook-Parallel",
                zh=zh_sample, en=en_sample,
                shared={"isbn": meta.get("isbn"), "year": meta.get("year")},
                alignment=AlignmentInfo(source="OpenLibrary+Douban", method="metadata_pair"),
            ))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", default="data/output/cover_book_parallel")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    build_cover_book_parallel(args.metadata, args.output, args.config)


if __name__ == "__main__":
    cli()
