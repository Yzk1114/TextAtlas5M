"""Build TextScenesHQ-CN.

Pipeline (mirrors paper §3.2 + §B.2 + §B.7):
1. Crawl images per topic from CommonCrawl-zh / Wukong-100M / Zero / Weibo public.
   We expect users to provide a JSONL of ``{"topic", "image_path"}`` records.
2. Run PaddleOCR PP-OCRv4 to keep images with ≥10 Chinese words.
3. Use Llama-3.1 / Qwen2.5 to fix spelling errors and reorder text JSON.
4. Use Qwen2.5-Coder to repair malformed JSON if needed.
5. Use Qwen2.5-VL to caption the *background* (without echoing in-image text).
6. Combine 500 GPT-4o-generated CN scene templates with the corrected text JSON
   to produce the unified prompt.
7. Cross-model agreement check (Qwen-VL vs InternVL2.5) for hard cases →
   forwarded to a human review queue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.llm import LLMClient
from ...common.ocr import ChineseOCR, sort_ocr_lines
from ...common.quality import ImageQualityFilter
from ...common.schema import OcrLine, TextAtlasSample
from ...common.templates import CaptionContext, load_templates, render_caption


SPELL_PROMPT = (
    "下面是一段从中文场景图像中识别出来的 OCR 文本，可能包含错别字、字母数字混淆或大小写错误。\n"
    "请基于上下文进行最小改动的修正，并保持原有顺序，最终输出修正后的文本：\n{ocr}"
)
REORDER_PROMPT = (
    "以下是按从左上到右下顺序拼接的 OCR 文本（由若干列组成），请根据语义重新整理为流畅的中文段落，"
    "若涉及多列布局请合并为正确阅读顺序。原文本：\n{ocr}"
)
JSON_REPAIR_PROMPT = (
    "请将下面的内容整理为合法 JSON，结构为 {{\"lines\": [{{\"text\": str, \"bbox\": [[x,y]*4]}}]}}。\n"
    "若无法解析的字段请保持空数组：\n{raw}"
)
BG_CAPTION_PROMPT = (
    "请用 80 字以内的中文描述这张图像的背景与主体场景，但 **不要** 复述图像中出现的任何文字。"
)
SCENE_TEMPLATE_PROMPT = (
    "请基于下列背景描述与文字 JSON，生成一段连贯的中文 prompt，使其既能描述场景，"
    "又能交代图中所有文字（保留原始拼写与顺序）：\n"
    "背景：{bg}\n文字 JSON：{text_json}"
)


def _ocr_to_lines(ocr_result) -> list[OcrLine]:
    return sort_ocr_lines(ocr_result.lines)


def build_text_scenes_hq_cn(
    records_jsonl: Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    min_words: int = 10,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    review_dir = Path(output_dir) / "human_review"
    for d in (image_dir, meta_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    ocr = ChineseOCR(primary=cfg["ocr"]["primary"])
    text_llm = LLMClient(provider=cfg["llm"]["default_provider"], model=cfg["llm"]["default_model"], cache_dir=cache_root)
    vlm_a = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"], cache_dir=cache_root)
    vlm_b = LLMClient(provider="dashscope", model="qwen2.5-vl-72b-instruct", cache_dir=cache_root)
    image_filter = ImageQualityFilter(
        ban_nsfw=cfg["quality"]["ban_nsfw"],
        ban_watermark=cfg["quality"]["ban_watermark"],
    )
    templates = load_templates()

    with JsonlShardWriter(meta_dir, "text_scenes_hq_cn", shard_size=cfg["export"]["shard_size"]) as writer, \
            records_jsonl.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_path = Path(rec["image_path"])
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            ocr_result = ocr.read(img)
            ocr_lines = _ocr_to_lines(ocr_result)
            words = "".join(l.text for l in ocr_lines)
            if len(words) < min_words:
                continue

            qf = image_filter.check(img_path)
            if qf.get("should_drop"):
                continue

            raw_text = "\n".join(l.text for l in ocr_lines)
            spelled = text_llm.chat(SPELL_PROMPT.format(ocr=raw_text), temperature=0.2, max_tokens=2048).text.strip()
            reordered = text_llm.chat(REORDER_PROMPT.format(ocr=spelled), temperature=0.3, max_tokens=2048).text.strip()
            text_json_str = text_llm.chat(
                JSON_REPAIR_PROMPT.format(raw=json.dumps(
                    [{"text": l.text, "bbox": l.bbox.points if l.bbox else []} for l in ocr_lines],
                    ensure_ascii=False,
                )),
                provider="dashscope", model="qwen2.5-coder-32b-instruct", temperature=0.0, max_tokens=4096,
            ).text.strip()
            try:
                text_json = json.loads(text_json_str)
            except Exception:
                text_json = {"lines": [{"text": l.text, "bbox": l.bbox.points if l.bbox else []} for l in ocr_lines]}

            sid = stable_id(rec.get("topic"), str(img_path))
            saved_path = save_image(img, image_dir, sid, fmt="jpg", quality=92)

            try:
                bg_caption_a = vlm_a.chat(BG_CAPTION_PROMPT, images=[saved_path], temperature=0.4, max_tokens=256).text.strip()
                bg_caption_b = vlm_b.chat(BG_CAPTION_PROMPT, images=[saved_path], temperature=0.4, max_tokens=256).text.strip()
            except Exception:
                bg_caption_a = bg_caption_b = ""

            divergence = _semantic_divergence(bg_caption_a, bg_caption_b)
            bg_caption = bg_caption_a if len(bg_caption_a) >= len(bg_caption_b) else bg_caption_b
            scene_prompt = ""
            if bg_caption:
                scene_prompt = text_llm.chat(
                    SCENE_TEMPLATE_PROMPT.format(bg=bg_caption, text_json=json.dumps(text_json, ensure_ascii=False)),
                    temperature=0.6, max_tokens=1024,
                ).text.strip()
            if not scene_prompt:
                scene_prompt = render_caption(CaptionContext(scene_caption=bg_caption or "中文场景", rendered_text=reordered), templates)

            sample = TextAtlasSample(
                sample_id=sid,
                image_path=saved_path,
                width=img.size[0], height=img.size[1],
                source_subset="TextScenesHQ-CN",
                layout_type="scene_hq",
                rendered_text=reordered,
                scene_caption=bg_caption,
                prompt=scene_prompt,
                ocr_lines=ocr_lines,
                metadata={
                    "topic": rec.get("topic"),
                    "ocr_raw": raw_text,
                    "ocr_spelled": spelled,
                    "text_json": text_json,
                    "vlm_caption_a": bg_caption_a,
                    "vlm_caption_b": bg_caption_b,
                    "divergence": divergence,
                    "needs_human_review": divergence > 0.5,
                    "image_quality": qf,
                },
            )
            if divergence > 0.5:
                # Forward to human review queue (copy to review_dir).
                (review_dir / f"{sid}.json").write_text(json.dumps(sample.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            writer.write(sample.to_dict())


def _semantic_divergence(a: str, b: str) -> float:
    """Cheap divergence: 1 - char-level Jaccard. 0 means identical."""
    if not a or not b:
        return 1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return 1.0 - inter / union


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", default="data/output/text_scenes_hq_cn")
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-words", type=int, default=10)
    args = parser.parse_args()
    build_text_scenes_hq_cn(args.records, args.output, args.config, args.min_words)


if __name__ == "__main__":
    cli()
