from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import DatasetSample, TextBlock, write_jsonl
from .text_utils import TextFilterConfig, deduplicate_texts, is_valid_long_text, normalize_text, stable_id


def sort_text_blocks_reading_order(blocks: list[TextBlock]) -> list[TextBlock]:
    """Sort OCR blocks top-to-bottom, then left-to-right."""

    sorted_blocks = sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]))
    for index, block in enumerate(sorted_blocks):
        block.reading_order = index
    return sorted_blocks


def _bbox_from_record(record: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = record.get("bbox") or record.get("box")
    if bbox and len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]

    polygon = record.get("polygon") or record.get("points")
    if polygon:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        return min(xs), min(ys), max(xs), max(ys)
    return 0.0, 0.0, 0.0, 0.0


def text_blocks_from_ocr_records(records: Iterable[dict[str, Any]]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for record in records:
        text = normalize_text(str(record.get("text") or record.get("transcription") or record.get("label") or ""))
        if not text:
            continue
        blocks.append(TextBlock(text=text, bbox=_bbox_from_record(record)))
    return sort_text_blocks_reading_order(blocks)


def sample_from_ocr_json(
    record: dict[str, Any],
    subset: str,
    text_filter: TextFilterConfig | None = None,
    split: str = "train",
) -> DatasetSample | None:
    """Convert an OCR-style JSON record to the unified dataset schema."""

    image_path = str(record.get("image_path") or record.get("image") or record.get("path") or "")
    ocr_records = record.get("text_blocks") or record.get("ocr") or record.get("annotations") or []
    blocks = text_blocks_from_ocr_records(ocr_records)
    combined_text = normalize_text(" ".join(block.text for block in blocks), keep_punctuation=False)
    if not is_valid_long_text(combined_text, text_filter):
        return None

    caption = normalize_text(str(record.get("caption") or record.get("prompt") or record.get("description") or ""))
    if caption:
        prompt = f"{caption}。图中包含以下中文文字：{combined_text}"
    else:
        prompt = f"生成一张包含清晰中文长文本的真实场景图片，文字内容包括：{combined_text}"

    sample_id = str(record.get("sample_id") or record.get("id") or stable_id(subset, image_path, combined_text, prefix="real_zh_"))
    return DatasetSample(
        sample_id=sample_id,
        subset=subset,
        image_path=image_path,
        prompt=prompt,
        split=split,  # type: ignore[arg-type]
        text_blocks=blocks,
        metadata={key: value for key, value in record.items() if key not in {"text_blocks", "ocr", "annotations"}},
    )


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    return [data]


def filter_ocr_dataset(
    input_path: str | Path,
    output_path: str | Path,
    subset: str = "LongWordsSubsetZH",
    text_filter: TextFilterConfig | None = None,
    deduplicate: bool = True,
) -> int:
    """Filter existing OCR datasets into a Chinese long-text subset."""

    records = load_json_or_jsonl(input_path)
    samples: list[DatasetSample] = []
    seen_texts: set[str] = set()
    for record in records:
        sample = sample_from_ocr_json(record, subset=subset, text_filter=text_filter)
        if sample is None:
            continue
        text = normalize_text(" ".join(block.text for block in sample.text_blocks), keep_punctuation=False)
        if deduplicate:
            # Fast exact-normalized dedup before optional near-duplicate pass.
            if text in seen_texts:
                continue
            seen_texts.add(text)
        samples.append(sample)

    if deduplicate and samples:
        texts = [" ".join(block.text for block in sample.text_blocks) for sample in samples]
        kept_texts = set(deduplicate_texts(texts))
        samples = [sample for sample, text in zip(samples, texts) if text in kept_texts]

    return write_jsonl(samples, output_path)


def filter_long_words_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    config: TextFilterConfig | None = None,
    subset: str = "LongWordsSubsetZH",
    deduplicate: bool = True,
) -> int:
    """Backward-compatible wrapper for filtering existing OCR JSON/JSONL data."""

    return filter_ocr_dataset(
        input_path=input_path,
        output_path=output_path,
        subset=subset,
        text_filter=config,
        deduplicate=deduplicate,
    )


filter_ocr_jsonl = filter_long_words_jsonl
