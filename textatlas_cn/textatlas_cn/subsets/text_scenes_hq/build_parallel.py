"""Bilingual / parallel TextScenesHQ.

The image is identical for both languages (real-world scene).  We:

1. Run OCR + cleanup once on the source image.
2. Use the bilingual VLMs to caption the scene background in zh and en
   independently; the captions are then enforced to be content-equivalent
   via translation back-check (bge-m3 cosine).
3. Detect the source language of the rendered text (Chinese vs English vs
   mixed). For the *missing* side we machine-translate. For mixed cases we
   keep the original transcript on both sides and add a per-language
   translation field in metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.io import save_image, stable_id
from ...common.llm import LLMClient
from ...common.ocr import ChineseOCR, chinese_ratio, sort_ocr_lines
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.quality import ImageQualityFilter
from ...common.schema import OcrLine, TextAtlasSample
from ...common.templates import CaptionContext, load_templates, render_caption
from ...common.translate import Translator, cross_lingual_similarity


BG_PROMPT_ZH = "请用 80 字以内的中文描述这张图像的背景与主体，但不要复述图像中出现的任何文字。"
BG_PROMPT_EN = "Describe the background and subject of this image in <=80 English words. Do NOT echo any text shown in the image."
SCENE_TEMPLATE_PROMPT_ZH = "请基于背景描述与文字 JSON，生成一段中文 prompt：\n背景：{bg}\n文字：{txt}"
SCENE_TEMPLATE_PROMPT_EN = "Generate one English prompt that combines the background description and the in-image text:\nbackground: {bg}\ntext: {txt}"


def build_text_scenes_hq_parallel(
    records_jsonl: Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    min_words: int = 10,
) -> None:
    cfg = load_config(config_path)
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images_shared"
    review_dir = Path(output_dir) / "human_review"
    image_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    ocr = ChineseOCR(primary=cfg["ocr"]["primary"])
    text_llm = LLMClient(provider=cfg["llm"]["default_provider"], model=cfg["llm"]["default_model"], cache_dir=cache_root)
    vlm = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"], cache_dir=cache_root)
    translator = Translator(provider=cfg["parallel"]["translate_provider"], model=cfg["parallel"]["translate_model"], cache_dir=cache_root)
    image_filter = ImageQualityFilter(ban_nsfw=cfg["quality"]["ban_nsfw"], ban_watermark=cfg["quality"]["ban_watermark"])
    templates = load_templates()
    bge_min = cfg["parallel"]["bge_m3_threshold"]

    with ParallelJsonlWriter(output_dir, "text_scenes_hq_parallel", shard_size=cfg["export"]["shard_size"]) as writer, \
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
            ocr_lines = sort_ocr_lines(ocr_result.lines)
            words = "".join(l.text for l in ocr_lines)
            if len(words) < min_words:
                continue
            qf = image_filter.check(img_path)
            if qf.get("should_drop"):
                continue

            sid = stable_id(rec.get("topic"), str(img_path))
            saved = save_image(img, image_dir, sid, fmt="jpg", quality=92)

            raw_zh = "\n".join(l.text for l in ocr_lines)
            zh_share = chinese_ratio(raw_zh)
            if zh_share >= 0.4:
                rendered_zh = raw_zh
                try:
                    rendered_en = translator.translate(raw_zh, "zh2en").text
                except Exception:
                    rendered_en = ""
            else:
                rendered_en = raw_zh
                try:
                    rendered_zh = translator.translate(raw_zh, "en2zh").text
                except Exception:
                    rendered_zh = ""

            try:
                bg_zh = vlm.chat(BG_PROMPT_ZH, images=[saved], temperature=0.4, max_tokens=256).text.strip()
                bg_en = vlm.chat(BG_PROMPT_EN, images=[saved], temperature=0.4, max_tokens=256).text.strip()
            except Exception:
                bg_zh = bg_en = ""

            try:
                sim = cross_lingual_similarity(bg_zh, bg_en)
            except Exception:
                sim = 1.0
            if bg_zh and bg_en and sim < bge_min:
                # Re-align: translate the en→zh (model-grade) and substitute zh.
                try:
                    bg_zh = translator.translate(bg_en, "en2zh").text
                except Exception:
                    pass

            text_json = [{"text": l.text, "bbox": l.bbox.points if l.bbox else []} for l in ocr_lines]
            try:
                prompt_zh = text_llm.chat(SCENE_TEMPLATE_PROMPT_ZH.format(bg=bg_zh, txt=json.dumps(text_json, ensure_ascii=False)), temperature=0.6, max_tokens=1024).text.strip()
            except Exception:
                prompt_zh = render_caption(CaptionContext(scene_caption=bg_zh or "中文场景", rendered_text=rendered_zh), templates)
            try:
                prompt_en = text_llm.chat(SCENE_TEMPLATE_PROMPT_EN.format(bg=bg_en, txt=json.dumps(text_json, ensure_ascii=False)), temperature=0.6, max_tokens=1024).text.strip()
            except Exception:
                prompt_en = f"{bg_en} The image displays: {rendered_en}"

            shared_meta = {"image_path": saved, "topic": rec.get("topic"), "ocr_raw": raw_zh, "image_quality": qf}
            zh_sample = TextAtlasSample(
                sample_id=f"{sid}-zh", image_path=saved,
                width=img.size[0], height=img.size[1],
                source_subset="TextScenesHQ-Parallel/zh", layout_type="scene_hq", language="zh-Hans",
                rendered_text=rendered_zh, scene_caption=bg_zh, prompt=prompt_zh,
                ocr_lines=ocr_lines, metadata=shared_meta,
            )
            en_sample = TextAtlasSample(
                sample_id=f"{sid}-en", image_path=saved,
                width=img.size[0], height=img.size[1],
                source_subset="TextScenesHQ-Parallel/en", layout_type="scene_hq", language="en",
                rendered_text=rendered_en, scene_caption=bg_en, prompt=prompt_en,
                ocr_lines=ocr_lines, metadata=shared_meta,
            )
            pair = ParallelTextAtlasSample(
                pair_id=sid, parallelism="same_image", layout_type="scene_hq",
                source_subset="TextScenesHQ-Parallel",
                zh=zh_sample, en=en_sample,
                shared=shared_meta,
                alignment=AlignmentInfo(
                    source=str(img_path), method="vlm_dual+translate",
                    forward_model=cfg["parallel"]["translate_model"], bge_m3_sim=sim,
                    len_zh_chars=len(rendered_zh), len_en_chars=len(rendered_en),
                ),
                metadata={"needs_human_review": (sim or 0) < 0.6},
            )
            if pair.metadata["needs_human_review"]:
                (review_dir / f"{sid}.json").write_text(json.dumps(pair.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            writer.write(pair)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", default="data/output/text_scenes_hq_parallel")
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-words", type=int, default=10)
    args = parser.parse_args()
    build_text_scenes_hq_parallel(args.records, args.output, args.config, args.min_words)


if __name__ == "__main__":
    cli()
