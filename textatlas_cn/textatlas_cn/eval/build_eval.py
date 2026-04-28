"""Construct TextAtlasEval-CN.

Stratified sampling (1000 each):
- CleanTextSynth-CN: 200 samples per length bin in {64,128,256,512,1024}.
- StyledTextSynth-CN: uniform over the 18 topics.
- TextScenesHQ-CN: uniform over the 26 topics.
- TextVisionBlend-CN: random sampling.

Each sample is then routed to a human-review CLI / Label Studio import file
where annotators verify caption faithfulness, OCR ground-truth and bbox quality.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..common.io import JsonlShardWriter, iter_jsonl


def _stratified_clean(samples: list[dict], rng: random.Random, n_total: int = 1000) -> list[dict]:
    by_bin = defaultdict(list)
    for s in samples:
        b = s["metadata"].get("length_bin")
        if b in (64, 128, 256, 512, 1024):
            by_bin[b].append(s)
    per_bin = n_total // 5
    out: list[dict] = []
    for b in (64, 128, 256, 512, 1024):
        rng.shuffle(by_bin[b])
        out.extend(by_bin[b][:per_bin])
    return out


def _stratified_topic(samples: list[dict], rng: random.Random, n_total: int) -> list[dict]:
    by_topic = defaultdict(list)
    for s in samples:
        topic = s["metadata"].get("topic", {}).get("name") or s["metadata"].get("topic")
        by_topic[topic].append(s)
    if not by_topic:
        rng.shuffle(samples)
        return samples[:n_total]
    per = max(1, n_total // len(by_topic))
    out: list[dict] = []
    for k, v in by_topic.items():
        rng.shuffle(v)
        out.extend(v[:per])
    rng.shuffle(out)
    return out[:n_total]


def build_textatlas_eval_cn(
    inputs: dict[str, Iterable[Path]],
    output_dir: str | Path,
    seed: int = 20260428,
) -> None:
    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = {
        "clean_text_synth_cn": ("CleanTextSynth-CN", _stratified_clean, 1000),
        "styled_text_synth_cn": ("StyledTextSynth-CN", _stratified_topic, 1000),
        "text_scenes_hq_cn": ("TextScenesHQ-CN", _stratified_topic, 1000),
        "text_vision_blend_cn": ("TextVisionBlend-CN", _stratified_topic, 1000),
    }

    review_path = out_dir / "human_review_queue.jsonl"
    with JsonlShardWriter(out_dir, "textatlas_eval_cn", shard_size=1000) as writer, \
            review_path.open("w", encoding="utf-8") as review_fh:
        for key, files in inputs.items():
            cfg = plans.get(key)
            if cfg is None:
                continue
            subset_name, strategy, n_total = cfg
            samples = []
            for fp in files:
                samples.extend(iter_jsonl(fp))
            samples = [s for s in samples if s["source_subset"] == subset_name]
            picked = strategy(samples, rng, n_total)
            for s in picked:
                s = dict(s)
                s["eval_split"] = "TextAtlasEval-CN"
                s["needs_human_review"] = True
                writer.write(s)
                review_fh.write(json.dumps({
                    "sample_id": s["sample_id"],
                    "image_path": s["image_path"],
                    "rendered_text": s["rendered_text"],
                    "prompt": s["prompt"],
                    "subset": subset_name,
                }, ensure_ascii=False) + "\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-text", nargs="*", default=[])
    parser.add_argument("--styled", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--blend", nargs="*", default=[])
    parser.add_argument("--output", default="data/output/textatlas_eval_cn")
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    inputs = {
        "clean_text_synth_cn": [Path(p) for p in args.clean_text],
        "styled_text_synth_cn": [Path(p) for p in args.styled],
        "text_scenes_hq_cn": [Path(p) for p in args.scenes],
        "text_vision_blend_cn": [Path(p) for p in args.blend],
    }
    build_textatlas_eval_cn(inputs, args.output, args.seed)


if __name__ == "__main__":
    cli()
