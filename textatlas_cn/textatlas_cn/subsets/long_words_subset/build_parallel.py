"""Bilingual / parallel LongWordsSubset.

The image is identical (real OCR sample); we only translate the OCR text
and the unified caption. Records may already be bilingual, e.g. when the
source dataset is **ICDAR2017-MLT**, **ICDAR2019-MLT**, **MTWI-2018**, or
**SROIE+ReCTS**, in which case we skip translation and use the provided
zh/en transcripts.

Input record fields (per line):
    {"image_path", "ocr_lines": [{"text", "bbox", "lang?"}], "caption_zh?",
     "caption_en?"}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import stable_id
from ...common.llm import LLMClient
from ...common.ocr import (
    chinese_ratio,
    clean_text,
    has_consecutive_repeat,
    sort_ocr_lines,
    unique_char_ratio,
)
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.schema import BBox, OcrLine, TextAtlasSample
from ...common.templates import CaptionContext, load_templates, render_caption
from ...common.translate import Translator


UNIFY_PROMPT_ZH = (
    "我有一段中文场景描述 T 和一段 OCR 文本 O，请生成一段自然连贯的中文段落，"
    "把 OCR 文本作为画面中的可见文字自然嵌入描述里。\nT = {scene_caption}\nO = {rendered_text}"
)
UNIFY_PROMPT_EN = (
    "I have a scene description T and OCR text O. Generate one natural, fluent English paragraph "
    "that weaves the OCR text into the scene description.\nT = {scene_caption}\nO = {rendered_text}"
)


def _filter(rec: dict, min_chars: int, min_unique: float, max_repeat: int):
    cleaned: list[OcrLine] = []
    raw = rec.get("ocr_lines", [])
    for line in raw:
        text = clean_text(line.get("text", ""))
        if not text or len(text) <= 1:
            continue
        bbox = BBox(points=[(float(x), float(y)) for x, y in line["bbox"]]) if isinstance(line.get("bbox"), list) else None
        cleaned.append(OcrLine(text=text, bbox=bbox))
    if not cleaned:
        return None, None
    cleaned = sort_ocr_lines(cleaned)
    joined = "\n".join(l.text for l in cleaned)
    if len(joined) < min_chars or unique_char_ratio(joined) < min_unique or has_consecutive_repeat(joined, max_repeat):
        return None, None
    return cleaned, joined


def build_long_words_subset_parallel(
    records_jsonl: Path,
    output_dir: str | Path,
    source_name: str = "ICDAR-MLT",
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    translator = Translator(provider=cfg["parallel"]["translate_provider"], model=cfg["parallel"]["translate_model"], cache_dir=cache_root)
    llm_zh = LLMClient(provider=cfg["llm"]["default_provider"], model=cfg["llm"]["default_model"], cache_dir=cache_root)
    llm_en = LLMClient(provider=cfg["llm"]["default_provider"], model=cfg["llm"]["default_model"], cache_dir=cache_root)
    templates_zh = load_templates()

    with ParallelJsonlWriter(output_dir, f"long_words_subset_parallel_{source_name.lower()}", shard_size=cfg["export"]["shard_size"]) as writer, \
            records_jsonl.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ocr_lines, joined = _filter(
                rec,
                min_chars=cfg["quality"]["min_text_chars"],
                min_unique=cfg["quality"]["min_unique_word_ratio"],
                max_repeat=cfg["quality"]["max_consecutive_repeat"],
            )
            if not ocr_lines:
                continue

            # Decide which side is the source language of the OCR text.
            zh_chars = chinese_ratio(joined)
            if zh_chars >= 0.4:
                rendered_zh = joined
                rendered_en = rec.get("ocr_text_en") or translator.translate(joined, "zh2en").text
            else:
                rendered_en = joined
                rendered_zh = rec.get("ocr_text_zh") or translator.translate(joined, "en2zh").text

            scene_zh = rec.get("caption_zh", "中文场景图像")
            scene_en = rec.get("caption_en", "A scene image with visible text.")
            try:
                prompt_zh = llm_zh.chat(UNIFY_PROMPT_ZH.format(scene_caption=scene_zh, rendered_text=rendered_zh), temperature=0.7, max_tokens=512).text.strip()
            except Exception:
                prompt_zh = render_caption(CaptionContext(scene_caption=scene_zh, rendered_text=rendered_zh), templates_zh)
            try:
                prompt_en = llm_en.chat(UNIFY_PROMPT_EN.format(scene_caption=scene_en, rendered_text=rendered_en), temperature=0.7, max_tokens=512).text.strip()
            except Exception:
                prompt_en = f"{scene_en} The image visibly displays: {rendered_en}"

            sid = stable_id(source_name, rec.get("image_path"), joined[:48])
            shared_meta = {"image_path": rec["image_path"], "source_dataset": source_name}
            zh_sample = TextAtlasSample(
                sample_id=f"{sid}-zh", image_path=rec["image_path"],
                width=rec.get("width", 0), height=rec.get("height", 0),
                source_subset=f"LongWordsSubset-Parallel-{source_name}/zh",
                layout_type="real_dense_text", language="zh-Hans",
                rendered_text=rendered_zh, scene_caption=scene_zh, prompt=prompt_zh,
                ocr_lines=ocr_lines, metadata=shared_meta,
            )
            en_sample = TextAtlasSample(
                sample_id=f"{sid}-en", image_path=rec["image_path"],
                width=rec.get("width", 0), height=rec.get("height", 0),
                source_subset=f"LongWordsSubset-Parallel-{source_name}/en",
                layout_type="real_dense_text", language="en",
                rendered_text=rendered_en, scene_caption=scene_en, prompt=prompt_en,
                ocr_lines=ocr_lines, metadata=shared_meta,
            )
            writer.write(ParallelTextAtlasSample(
                pair_id=sid, parallelism="same_image", layout_type="real_dense_text",
                source_subset=f"LongWordsSubset-Parallel-{source_name}",
                zh=zh_sample, en=en_sample,
                shared=shared_meta,
                alignment=AlignmentInfo(source=source_name, method="dataset_pair_or_translate", len_zh_chars=len(rendered_zh), len_en_chars=len(rendered_en)),
            ))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--source", default="ICDAR-MLT")
    parser.add_argument("--output", default="data/output/long_words_subset_parallel")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    build_long_words_subset_parallel(args.records, args.output, args.source, args.config)


if __name__ == "__main__":
    cli()
