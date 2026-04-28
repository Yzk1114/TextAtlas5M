"""Build CoverBook-CN.

Input: a CSV/JSONL of book metadata + cover URL collected from public sources
(豆瓣读书、当当、OpenLibrary-zh、ISBN 数据库). We do **not** scrape inline; users
provide the metadata file. Each row must contain at least:

    {"title", "author", "publisher", "year", "category", "cover_url"}

Output samples mirror the paper's CoverBook prompt: ``"<title>｜作者：<author>｜出版：<publisher> <year>｜类别：<category>"``.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image
from tqdm import tqdm

from ...common.config import load_config
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.schema import OcrLine, TextAtlasSample


def _make_prompt(meta: dict) -> str:
    parts = [meta.get("title", "")]
    for key, prefix in [("author", "作者"), ("publisher", "出版社"), ("year", "出版年"), ("category", "类别"), ("blurb", "简介")]:
        if meta.get(key):
            parts.append(f"{prefix}：{meta[key]}")
    return "一本中文图书的封面，包含以下信息：" + " | ".join(p for p in parts if p)


def build_cover_book_cn(
    metadata_jsonl: Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with JsonlShardWriter(meta_dir, "cover_book_cn", shard_size=cfg["export"]["shard_size"]) as writer, \
            metadata_jsonl.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh):
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            url = meta.get("cover_url")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                continue

            sid = stable_id(meta.get("isbn") or meta.get("title"), meta.get("author"))
            img_path = save_image(img, image_dir, sid, fmt="jpg", quality=92)

            rendered = " ".join(filter(None, [meta.get("title"), meta.get("author"), meta.get("publisher")]))
            sample = TextAtlasSample(
                sample_id=sid,
                image_path=img_path,
                width=img.size[0], height=img.size[1],
                source_subset="CoverBook-CN",
                layout_type="cover_book",
                rendered_text=rendered,
                scene_caption="中文图书封面",
                prompt=_make_prompt(meta),
                ocr_lines=[OcrLine(text=rendered, bbox=None)] if rendered else [],
                metadata={"book": meta},
            )
            writer.write(sample.to_dict())


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", default="data/output/cover_book_cn")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    build_cover_book_cn(args.metadata, args.output, args.config)


if __name__ == "__main__":
    cli()
