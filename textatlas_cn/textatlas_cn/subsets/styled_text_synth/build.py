"""Build StyledTextSynth-CN.

Steps (per topic):
1. Use ``GENERAL_SCENE_PROMPT`` (or seed/template variant) to ask an LLM for
   a Chinese scene description.
2. Generate a text-free image with the configured T2I model.
3. Run :class:`TextRegionDetector` to obtain a fillable bbox/quad.
4. Generate Chinese text via LLM (or VLM for visual-dependent topics like
   cinema posters), deduplicate semantically.
5. Render text into the bbox / quad, optionally with perspective transform.
6. Build a unified caption with ``configs/topics_styled.yaml`` metadata.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.fonts import sample_font
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.llm import LLMClient
from ...common.quality import SemanticDeduper, TextQualityConfig, passes_text_quality
from ...common.render import render_text_in_quad, render_text_in_rect
from ...common.schema import OcrLine, TextAtlasSample
from ...common.t2i import T2IClient, T2IRequest
from ...common.templates import CaptionContext, load_templates, render_caption
from ...common.text_detect import DetectorConfig, TextRegionDetector
from .prompts import (
    GENERAL_SCENE_PROMPT,
    SEED_SCENE_PROMPT,
    TEXT_FOR_TOPIC_PROMPT_GPT,
    VLM_TEXT_FOR_IMAGE_PROMPT,
)


def _split_numbered(text: str) -> list[str]:
    items = re.split(r"\n?\s*\d+\.[\s]?", text)
    return [it.strip() for it in items if it.strip()]


def _scene_prompt(topic_cfg: dict[str, Any]) -> str:
    cn = topic_cfg["cn"]
    if topic_cfg["prompt_type"] == "seed":
        return SEED_SCENE_PROMPT.format(seed=topic_cfg["seed"])
    return GENERAL_SCENE_PROMPT.format(topic=cn)


def _generate_text_pool(llm: LLMClient, topic_cfg: dict[str, Any], n: int = 50) -> list[str]:
    if topic_cfg.get("text_provider") == "vlm":
        return []
    min_len, max_len = topic_cfg.get("word_count", [40, 80])
    prompt = TEXT_FOR_TOPIC_PROMPT_GPT.format(
        n=n, topic=topic_cfg["cn"], min_len=min_len, max_len=max_len
    )
    resp = llm.chat(prompt, temperature=0.9, max_tokens=4096)
    return _split_numbered(resp.text)


def _generate_text_from_image(llm: LLMClient, topic_cfg: dict[str, Any], image_path: str) -> str:
    prompt = VLM_TEXT_FOR_IMAGE_PROMPT.format(topic=topic_cfg["cn"])
    resp = llm.chat(prompt, images=[image_path], temperature=0.7, max_tokens=512)
    try:
        data = json.loads(resp.text.strip().strip("`"))
        title, body = data.get("title", ""), data.get("body", "")
        return f"{title}\n{body}".strip()
    except Exception:
        return resp.text.strip()


def build_styled_text_synth_cn(
    topics_yaml: str | Path,
    per_topic: int,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    seed: int = 20260428,
) -> None:
    cfg = load_config(config_path)
    fonts_root = resolve_path(cfg, "fonts_root")
    cache_root = resolve_path(cfg, "cache_root")
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    text_llm = LLMClient(provider=cfg["llm"]["default_provider"], model=cfg["llm"]["default_model"], cache_dir=cache_root)
    vision_llm = LLMClient(provider=cfg["vlm"]["default_provider"], model=cfg["vlm"]["default_model"], cache_dir=cache_root)
    t2i = T2IClient(provider=cfg["t2i"]["default_provider"], model=cfg["t2i"]["default_model"])

    detector = TextRegionDetector(DetectorConfig(yolo_weights=cfg.get("detector", {}).get("yolo_weights", "data/models/yolo11l_textregion.pt")))
    deduper = SemanticDeduper()
    quality_cfg = TextQualityConfig(
        min_chinese_ratio=cfg["quality"]["min_chinese_ratio"],
        min_unique_word_ratio=cfg["quality"]["min_unique_word_ratio"],
        max_consecutive_repeat=cfg["quality"]["max_consecutive_repeat"],
        min_text_chars=cfg["quality"]["min_text_chars"],
    )
    templates = load_templates()

    topics = yaml.safe_load(Path(topics_yaml).read_text(encoding="utf-8"))["topics"]

    with JsonlShardWriter(meta_dir, "styled_text_synth_cn", shard_size=cfg["export"]["shard_size"]) as writer:
        for topic_cfg in topics:
            text_pool = _generate_text_pool(text_llm, topic_cfg) if topic_cfg.get("text_provider") != "vlm" else []
            text_pool = [t for t in text_pool if passes_text_quality(t, quality_cfg) and not deduper.is_duplicate(t)]
            produced = 0
            attempt_budget = per_topic * 3
            attempts = 0
            while produced < per_topic and attempts < attempt_budget:
                attempts += 1
                scene_prompt_template = _scene_prompt(topic_cfg)
                resp = text_llm.chat(scene_prompt_template, temperature=0.9, max_tokens=512)
                scene_caption = resp.text.strip()
                req = T2IRequest(
                    prompt=scene_caption,
                    width=cfg["t2i"]["width"], height=cfg["t2i"]["height"],
                    steps=cfg["t2i"]["steps"], guidance_scale=cfg["t2i"]["guidance_scale"],
                    seed=rng.randint(0, 2**31 - 1),
                )
                try:
                    raw_img = t2i.generate(req)
                except Exception:
                    continue

                bbox = detector.detect(raw_img, group=topic_cfg["detector_group"])
                if bbox is None:
                    continue

                if topic_cfg.get("text_provider") == "vlm":
                    sid_tmp = stable_id("vlm-text", topic_cfg["name"], attempts)
                    tmp_path = save_image(raw_img, image_dir / "_tmp", sid_tmp)
                    text = _generate_text_from_image(vision_llm, topic_cfg, tmp_path)
                else:
                    if not text_pool:
                        break
                    text = text_pool.pop()

                if not passes_text_quality(text, quality_cfg):
                    continue

                font = sample_font(fonts_root, rng=rng)
                try:
                    is_quad = len(bbox.points) == 4 and not _is_axis_aligned(bbox)
                    if topic_cfg["render_strategy"] == "template" or not is_quad:
                        rendered, font_attrs = render_text_in_rect(raw_img, bbox, text, font["path"])
                    else:
                        rendered, font_attrs = render_text_in_quad(raw_img, bbox, text, font["path"])
                except Exception:
                    continue

                sid = stable_id(topic_cfg["name"], scene_caption[:64], text[:64])
                img_path = save_image(rendered, image_dir, sid)
                ctx = CaptionContext(scene_caption=scene_caption, rendered_text=text)
                final_prompt = render_caption(ctx, templates, rng)

                sample = TextAtlasSample(
                    sample_id=sid,
                    image_path=img_path,
                    width=rendered.size[0], height=rendered.size[1],
                    source_subset="StyledTextSynth-CN",
                    layout_type="styled_scene",
                    rendered_text=text,
                    scene_caption=scene_caption,
                    prompt=final_prompt,
                    ocr_lines=[OcrLine(text=text, bbox=bbox, font=font_attrs)],
                    metadata={
                        "topic": topic_cfg,
                        "font": font,
                        "render_strategy": topic_cfg["render_strategy"],
                    },
                )
                writer.write(sample.to_dict())
                produced += 1


def _is_axis_aligned(bbox) -> bool:
    xs = [p[0] for p in bbox.points]
    ys = [p[1] for p in bbox.points]
    return len(set(round(x) for x in xs)) <= 2 and len(set(round(y) for y in ys)) <= 2


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--per-topic", type=int, required=True)
    parser.add_argument("--output", default="data/output/styled_text_synth_cn")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    build_styled_text_synth_cn(args.topics, args.per_topic, args.output, args.config, args.seed)


if __name__ == "__main__":
    cli()
