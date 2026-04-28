"""Build LongWordsSubset-CN.

The paper's LongWordsSubset is built by filtering AnyWords3M / Marion10M for
long English captions. The Chinese counterpart filters Chinese OCR datasets:
    - AnyWord-CN (AnyText 项目中文部分)
    - RCTW-17, ICDAR-LSVT, ICDAR-ReCTS, MTWI-2018, ICDAR2017-MLT (zh)

Each input record is expected to provide:
    {"image_path", "ocr_lines": [{"text", "bbox"}], "caption"}

Filters (mirroring §B.2.1 with Chinese adaptations):
1. min character count (≥ ``min_text_chars`` Chinese chars across all lines)
2. unique-character ratio > ``min_unique_word_ratio``
3. no >3 consecutive identical chars
4. each line must have ≥1 CJK char and length > 1
5. text cleaning (drop non-CJK/digits/punctuation noise)
6. spatial sort top→bottom, left→right
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from ...common.config import load_config
from ...common.io import JsonlShardWriter, stable_id
from ...common.llm import LLMClient
from ...common.ocr import (
    chinese_ratio,
    clean_text,
    has_consecutive_repeat,
    sort_ocr_lines,
    unique_char_ratio,
)
from ...common.schema import BBox, OcrLine, TextAtlasSample
from ...common.templates import CaptionContext, load_templates, render_caption


UNIFY_PROMPT = (
    "我有一段中文场景描述 T 和一段 OCR 文本 O，请生成一段自然、连贯的中文段落，"
    "要求把 OCR 文本作为画面中的可见文字自然嵌入描述里，不要刻意罗列。\n"
    "T = {scene_caption}\nO = {rendered_text}"
)


def _filter_record(rec: dict, min_chars: int = 7, min_unique: float = 0.3, max_repeat: int = 3) -> tuple[bool, list[OcrLine], str]:
    raw_lines = rec.get("ocr_lines", [])
    cleaned: list[OcrLine] = []
    for line in raw_lines:
        text = clean_text(line.get("text", ""))
        if not text or len(text) <= 1:
            continue
        if chinese_ratio(text) < 0.2:
            continue
        bbox = BBox(points=[(float(x), float(y)) for x, y in line["bbox"]]) if isinstance(line.get("bbox"), list) else None
        cleaned.append(OcrLine(text=text, bbox=bbox))

    if not cleaned:
        return False, [], ""
    cleaned = sort_ocr_lines(cleaned)
    joined = "\n".join(l.text for l in cleaned)
    if len(joined) < min_chars:
        return False, [], ""
    if unique_char_ratio(joined) < min_unique:
        return False, [], ""
    if has_consecutive_repeat(joined, max_repeat):
        return False, [], ""
    return True, cleaned, joined


def build_long_words_subset_cn(
    records_jsonl: Path,
    output_dir: str | Path,
    source_name: str = "AnyWordCN",
    config_path: str | Path | None = None,
    use_llm_unify: bool = True,
) -> None:
    cfg = load_config(config_path)
    meta_dir = Path(output_dir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    templates = load_templates()
    llm = (
        LLMClient(provider=cfg["llm"]["default_provider"], model=cfg["llm"]["default_model"])
        if use_llm_unify
        else None
    )

    with JsonlShardWriter(meta_dir, f"long_words_subset_cn_{source_name.lower()}", shard_size=cfg["export"]["shard_size"]) as writer, \
            records_jsonl.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ok, ocr_lines, joined = _filter_record(
                rec,
                min_chars=cfg["quality"]["min_text_chars"],
                min_unique=cfg["quality"]["min_unique_word_ratio"],
                max_repeat=cfg["quality"]["max_consecutive_repeat"],
            )
            if not ok:
                continue
            scene_caption = rec.get("caption", "")
            prompt = ""
            if llm is not None and scene_caption:
                try:
                    prompt = llm.chat(UNIFY_PROMPT.format(scene_caption=scene_caption, rendered_text=joined),
                                      temperature=0.7, max_tokens=512).text.strip()
                except Exception:
                    prompt = ""
            if not prompt:
                prompt = render_caption(CaptionContext(scene_caption=scene_caption or "中文场景图像", rendered_text=joined), templates)

            sid = stable_id(source_name, rec.get("image_path") or rec.get("id"), joined[:64])
            sample = TextAtlasSample(
                sample_id=sid,
                image_path=rec["image_path"],
                width=rec.get("width", 0),
                height=rec.get("height", 0),
                source_subset=f"LongWordsSubset-CN-{source_name}",
                layout_type="real_dense_text",
                rendered_text=joined,
                scene_caption=scene_caption,
                prompt=prompt,
                ocr_lines=ocr_lines,
                metadata={"original_meta": rec.get("meta", {})},
            )
            writer.write(sample.to_dict())


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", default="data/output/long_words_subset_cn")
    parser.add_argument("--source", default="AnyWordCN")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    build_long_words_subset_cn(args.records, args.output, args.source, args.config, use_llm_unify=not args.no_llm)


if __name__ == "__main__":
    cli()
