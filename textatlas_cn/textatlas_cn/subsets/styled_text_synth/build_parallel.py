"""Bilingual / parallel StyledTextSynth.

Pipeline:
1. Pick a topic from ``configs/topics_styled.yaml``.
2. Use an LLM to generate one **bilingual** scene description (zh + en); the
   English description is used as the T2I prompt so that the synthesized
   base image is identical for both languages.
3. Run the configured T2I model **once** to produce the no-text base image.
4. Detect the fillable region with YOLO/SAM2 (shared bbox).
5. Generate Chinese text with the topic's normal generator, and translate
   it to English (or generate both in one call). Quality-filter both sides.
6. Render text into the same bbox twice (one zh font, one en font) and emit
   the parallel pair.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.font_pairs import sample_font_pair
from ...common.io import save_image, stable_id
from ...common.llm import LLMClient
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.quality import SemanticDeduper, TextQualityConfig, passes_text_quality
from ...common.render import render_text_in_quad, render_text_in_rect
from ...common.schema import OcrLine, TextAtlasSample
from ...common.t2i import T2IClient, T2IRequest
from ...common.templates import CaptionContext, load_templates, render_caption
from ...common.text_detect import DetectorConfig, TextRegionDetector
from ...common.translate import Translator, cross_lingual_similarity


SCENE_PROMPT_BILINGUAL = (
    "I want to render a parallel zh/en image about {topic_en} ({topic_cn}). "
    "Output ONE JSON object with two scene descriptions for an image generator: "
    "`scene_en` (≤ 160 English words) and `scene_zh` (≤ 160 Chinese characters). "
    "Both descriptions MUST refer to the same scene with identical layout, camera, "
    "lighting, background and subject; the {topic_en} should face the camera, take "
    "≥1/3 of the image, contain NO visible text, and have a consistent colour. "
    "Return only the JSON, with double quotes."
)

TEXT_PROMPT_BILINGUAL = (
    "Generate {n} parallel zh/en text snippets for a {topic_en} ({topic_cn}). "
    "Each snippet must be content-equivalent across the two languages. "
    "Constraint: zh = {min_len}~{max_len} Chinese characters; en = the natural translation "
    "(do NOT pad or truncate). "
    "Return a JSON array of objects with keys `zh` and `en`. Output JSON only."
)


def _scene_pair(llm: LLMClient, topic_cfg: dict[str, Any]) -> tuple[str, str]:
    prompt = SCENE_PROMPT_BILINGUAL.format(topic_en=topic_cfg["name"].replace("_", " "), topic_cn=topic_cfg["cn"])
    resp = llm.chat(prompt, temperature=0.8, max_tokens=1024)
    raw = resp.text.strip().strip("`")
    try:
        data = json.loads(raw)
        return data.get("scene_en", "").strip(), data.get("scene_zh", "").strip()
    except Exception:
        # fall back: ask separately
        en = llm.chat(f"Describe a {topic_cfg['name']} scene without text in <=160 words.", temperature=0.7, max_tokens=512).text
        zh = llm.chat(f"用不超过160字的中文描述一个{topic_cfg['cn']}场景，且不包含任何文字。", temperature=0.7, max_tokens=512).text
        return en.strip(), zh.strip()


def _text_pairs(llm: LLMClient, topic_cfg: dict[str, Any], n: int = 50) -> list[tuple[str, str]]:
    min_len, max_len = topic_cfg.get("word_count", [40, 80])
    prompt = TEXT_PROMPT_BILINGUAL.format(
        n=n, topic_en=topic_cfg["name"].replace("_", " "), topic_cn=topic_cfg["cn"],
        min_len=min_len, max_len=max_len,
    )
    resp = llm.chat(prompt, temperature=0.9, max_tokens=4096)
    raw = resp.text.strip().strip("`")
    try:
        data = json.loads(raw)
        return [(item["zh"].strip(), item["en"].strip()) for item in data if item.get("zh") and item.get("en")]
    except Exception:
        return []


def build_styled_text_synth_parallel(
    topics_yaml: str | Path,
    per_topic: int,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    seed: int = 20260428,
) -> None:
    cfg = load_config(config_path)
    fonts_root = resolve_path(cfg, "fonts_root")
    en_fonts_root = resolve_path(cfg, "fonts_en_root")
    cache_root = resolve_path(cfg, "cache_root")
    img_zh = Path(output_dir) / "images_zh"
    img_en = Path(output_dir) / "images_en"
    img_zh.mkdir(parents=True, exist_ok=True)
    img_en.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    text_llm = LLMClient(provider=cfg["llm"]["default_provider"], model=cfg["llm"]["default_model"], cache_dir=cache_root)
    t2i = T2IClient(provider=cfg["t2i"]["default_provider"], model=cfg["t2i"]["default_model"])
    translator = Translator(provider=cfg["parallel"]["translate_provider"], model=cfg["parallel"]["translate_model"], cache_dir=cache_root)
    detector = TextRegionDetector(DetectorConfig(yolo_weights=cfg.get("detector", {}).get("yolo_weights", "data/models/yolo11l_textregion.pt")))
    deduper_zh = SemanticDeduper()
    deduper_en = SemanticDeduper(model_name="BAAI/bge-base-en-v1.5")
    quality = TextQualityConfig(
        min_chinese_ratio=cfg["quality"]["min_chinese_ratio"],
        min_unique_word_ratio=cfg["quality"]["min_unique_word_ratio"],
        max_consecutive_repeat=cfg["quality"]["max_consecutive_repeat"],
        min_text_chars=cfg["quality"]["min_text_chars"],
    )
    templates = load_templates()

    topics = yaml.safe_load(Path(topics_yaml).read_text(encoding="utf-8"))["topics"]
    bge_min = cfg["parallel"]["bge_m3_threshold"]

    with ParallelJsonlWriter(output_dir, "styled_text_synth_parallel", shard_size=cfg["export"]["shard_size"]) as writer:
        for topic_cfg in topics:
            pairs_pool = _text_pairs(text_llm, topic_cfg, n=per_topic * 2)
            # If the LLM didn't return JSON pairs, fall back to monolingual then translate.
            if not pairs_pool:
                fallback_zh = text_llm.chat(
                    f"请生成 {per_topic*2} 段关于 {topic_cfg['cn']} 的中文文字，每段 40-80 字，输出格式 1.xxx 2.xxx ...",
                    temperature=0.9, max_tokens=4096,
                ).text
                items = [t.strip() for t in fallback_zh.split("\n") if t.strip()]
                pairs_pool = []
                for it in items:
                    en = translator.translate(it, direction="zh2en").text
                    pairs_pool.append((it, en))

            produced, attempts, budget = 0, 0, per_topic * 4
            while produced < per_topic and attempts < budget:
                attempts += 1
                if not pairs_pool:
                    break
                zh_text, en_text = pairs_pool.pop()
                if not passes_text_quality(zh_text, quality):
                    continue
                if deduper_zh.is_duplicate(zh_text) or deduper_en.is_duplicate(en_text):
                    continue
                # Cross-lingual similarity check.
                try:
                    sim = cross_lingual_similarity(zh_text, en_text)
                except Exception:
                    sim = 1.0
                if sim < bge_min:
                    continue

                scene_en, scene_zh = _scene_pair(text_llm, topic_cfg)
                req = T2IRequest(
                    prompt=scene_en,
                    width=cfg["t2i"]["width"], height=cfg["t2i"]["height"],
                    steps=cfg["t2i"]["steps"], guidance_scale=cfg["t2i"]["guidance_scale"],
                    seed=rng.randint(0, 2**31 - 1),
                )
                try:
                    base_img = t2i.generate(req)
                except Exception:
                    continue
                bbox = detector.detect(base_img, group=topic_cfg["detector_group"])
                if bbox is None:
                    continue

                try:
                    zh_font, en_font = sample_font_pair(fonts_root, en_fonts_root, rng=rng)
                except FileNotFoundError:
                    return

                try:
                    is_quad = len(bbox.points) == 4 and not _is_axis_aligned(bbox)
                    render_fn = (render_text_in_rect if topic_cfg["render_strategy"] == "template" or not is_quad else render_text_in_quad)
                    zh_img, zh_attrs = render_fn(base_img, bbox, zh_text, zh_font["path"])
                    en_img, en_attrs = render_fn(base_img, bbox, en_text, en_font["path"])
                except Exception:
                    continue

                pair_id = stable_id("StyledText", topic_cfg["name"], scene_en[:48], en_text[:48])
                zh_path = save_image(zh_img, img_zh, pair_id)
                en_path = save_image(en_img, img_en, pair_id)

                ctx_zh = CaptionContext(scene_caption=scene_zh, rendered_text=zh_text)
                ctx_en = CaptionContext(scene_caption=scene_en, rendered_text=en_text)
                zh_prompt = render_caption(ctx_zh, templates, rng)
                en_prompt = f"{scene_en} The image must clearly render: {en_text}"

                zh_sample = TextAtlasSample(
                    sample_id=f"{pair_id}-zh", image_path=zh_path,
                    width=zh_img.size[0], height=zh_img.size[1],
                    source_subset="StyledTextSynth-Parallel/zh",
                    layout_type="styled_scene", language="zh-Hans",
                    rendered_text=zh_text, scene_caption=scene_zh, prompt=zh_prompt,
                    ocr_lines=[OcrLine(text=zh_text, bbox=bbox, font=zh_attrs)],
                    metadata={"topic": topic_cfg, "font": zh_font, "render_strategy": topic_cfg["render_strategy"]},
                )
                en_sample = TextAtlasSample(
                    sample_id=f"{pair_id}-en", image_path=en_path,
                    width=en_img.size[0], height=en_img.size[1],
                    source_subset="StyledTextSynth-Parallel/en",
                    layout_type="styled_scene", language="en",
                    rendered_text=en_text, scene_caption=scene_en, prompt=en_prompt,
                    ocr_lines=[OcrLine(text=en_text, bbox=bbox, font=en_attrs)],
                    metadata={"topic": topic_cfg, "font": en_font, "render_strategy": topic_cfg["render_strategy"]},
                )
                writer.write(ParallelTextAtlasSample(
                    pair_id=pair_id,
                    parallelism="shared_layout",
                    layout_type="styled_scene",
                    source_subset="StyledTextSynth-Parallel",
                    zh=zh_sample, en=en_sample,
                    shared={
                        "topic": topic_cfg,
                        "scene_prompt_en": scene_en,
                        "scene_prompt_zh": scene_zh,
                        "bbox": bbox.points,
                        "render_strategy": topic_cfg["render_strategy"],
                        "font_pair_style": zh_font.get("shared_style"),
                    },
                    alignment=AlignmentInfo(
                        source="GPT4o-bilingual" if cfg["llm"]["default_provider"] == "openai" else cfg["llm"]["default_model"],
                        method="llm_translate",
                        forward_model=cfg["llm"]["default_model"],
                        bge_m3_sim=sim,
                        len_zh_chars=len(zh_text), len_en_chars=len(en_text),
                    ),
                ))
                produced += 1


def _is_axis_aligned(bbox) -> bool:
    xs = [p[0] for p in bbox.points]
    ys = [p[1] for p in bbox.points]
    return len(set(round(x) for x in xs)) <= 2 and len(set(round(y) for y in ys)) <= 2


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--per-topic", type=int, required=True)
    parser.add_argument("--output", default="data/output/styled_text_synth_parallel")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    build_styled_text_synth_parallel(args.topics, args.per_topic, args.output, args.config, args.seed)


if __name__ == "__main__":
    cli()
