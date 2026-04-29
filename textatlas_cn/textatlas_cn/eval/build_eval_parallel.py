"""Construct TextAtlasEval-Parallel (zh/en).

Reads parallel JSONL shards (one row per pair) and stratifies the same way
as the monolingual eval builder, but at the **pair** level so that the zh
and en splits are guaranteed to be aligned by pair_id.

For CleanTextSynth-Parallel, the stratum key is the English ``length_bin``
attached to ``shared``/``alignment.bin_anchor``; for the other subsets we
stratify by topic (Styled / Scenes) or random (Blend), exactly mirroring
the original paper's procedure.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..common.io import iter_jsonl


def _stratified_clean(pairs: list[dict], rng: random.Random, n_total: int = 1000) -> list[dict]:
    by_bin = defaultdict(list)
    for p in pairs:
        b = p.get("alignment", {}).get("bin_anchor")
        if b in {"64", "128", "256", "512", "1024"}:
            by_bin[b].append(p)
    per_bin = n_total // 5
    out: list[dict] = []
    for b in ("64", "128", "256", "512", "1024"):
        rng.shuffle(by_bin[b])
        out.extend(by_bin[b][:per_bin])
    return out


def _stratified_topic(pairs: list[dict], rng: random.Random, n_total: int) -> list[dict]:
    by_topic = defaultdict(list)
    for p in pairs:
        topic = (p.get("shared", {}).get("topic") or {}).get("name") if isinstance(p.get("shared", {}).get("topic"), dict) else p.get("shared", {}).get("topic")
        by_topic[topic].append(p)
    if not by_topic:
        rng.shuffle(pairs)
        return pairs[:n_total]
    per = max(1, n_total // len(by_topic))
    out: list[dict] = []
    for k, v in by_topic.items():
        rng.shuffle(v)
        out.extend(v[:per])
    rng.shuffle(out)
    return out[:n_total]


def build_textatlas_eval_parallel(
    inputs: dict[str, Iterable[Path]],
    output_dir: str | Path,
    seed: int = 20260428,
) -> None:
    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = {
        "clean_text_synth_parallel": ("CleanTextSynth-Parallel", _stratified_clean, 1000),
        "styled_text_synth_parallel": ("StyledTextSynth-Parallel", _stratified_topic, 1000),
        "text_scenes_hq_parallel": ("TextScenesHQ-Parallel", _stratified_topic, 1000),
        "text_vision_blend_parallel": ("TextVisionBlend-Parallel", _stratified_topic, 1000),
    }

    parallel_path = out_dir / "textatlas_eval_parallel.jsonl"
    zh_path = out_dir / "textatlas_eval_zh.jsonl"
    en_path = out_dir / "textatlas_eval_en.jsonl"
    review_path = out_dir / "human_review_queue.jsonl"

    with parallel_path.open("w", encoding="utf-8") as pf, \
            zh_path.open("w", encoding="utf-8") as zf, \
            en_path.open("w", encoding="utf-8") as ef, \
            review_path.open("w", encoding="utf-8") as rf:
        for key, files in inputs.items():
            cfg = plans.get(key)
            if cfg is None:
                continue
            subset_name, strategy, n_total = cfg
            pairs: list[dict] = []
            for fp in files:
                pairs.extend(iter_jsonl(fp))
            pairs = [p for p in pairs if p["source_subset"] == subset_name]
            picked = strategy(pairs, rng, n_total)
            for p in picked:
                p = dict(p)
                p["eval_split"] = "TextAtlasEval-Parallel"
                p["needs_human_review"] = True
                pf.write(json.dumps(p, ensure_ascii=False) + "\n")
                zh_row = dict(p["zh"]); zh_row["pair_id"] = p["pair_id"]; zh_row["lang"] = "zh"
                en_row = dict(p["en"]); en_row["pair_id"] = p["pair_id"]; en_row["lang"] = "en"
                zf.write(json.dumps(zh_row, ensure_ascii=False) + "\n")
                ef.write(json.dumps(en_row, ensure_ascii=False) + "\n")
                rf.write(json.dumps({
                    "pair_id": p["pair_id"], "subset": subset_name,
                    "image_zh": p["zh"]["image_path"], "image_en": p["en"]["image_path"],
                    "rendered_zh": p["zh"]["rendered_text"], "rendered_en": p["en"]["rendered_text"],
                    "prompt_zh": p["zh"]["prompt"], "prompt_en": p["en"]["prompt"],
                }, ensure_ascii=False) + "\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-text", nargs="*", default=[])
    parser.add_argument("--styled", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--blend", nargs="*", default=[])
    parser.add_argument("--output", default="data/output/textatlas_eval_parallel")
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    inputs = {
        "clean_text_synth_parallel": [Path(p) for p in args.clean_text],
        "styled_text_synth_parallel": [Path(p) for p in args.styled],
        "text_scenes_hq_parallel": [Path(p) for p in args.scenes],
        "text_vision_blend_parallel": [Path(p) for p in args.blend],
    }
    build_textatlas_eval_parallel(inputs, args.output, args.seed)


if __name__ == "__main__":
    cli()
