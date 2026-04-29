"""Parallel zh↔en corpus loaders.

Each loader yields ``(en_text, zh_text, source_name)`` tuples. Streams from
HuggingFace if available, otherwise reads local TSV/JSONL fallbacks.

The mixed iterator weights corpora according to ``sampling_weight`` and
filters out pairs that fail ``decide_paired_lengths`` (e.g. zh translation
suspiciously shorter/longer than expected).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import yaml

from .length_bins import decide_paired_lengths


REPO_ROOT = Path(__file__).resolve().parents[2]
PARALLEL_YAML = REPO_ROOT / "configs" / "parallel_corpora.yaml"


def load_registry(yaml_path: Path | None = None) -> dict:
    p = Path(yaml_path or PARALLEL_YAML)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _resolve_field(item: dict, field_path: str) -> str | None:
    obj: object = item
    for part in field_path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj if isinstance(obj, str) else None


def iter_text_corpus(name: str, max_samples: int | None = None) -> Iterator[tuple[str, str, str]]:
    spec = next(c for c in load_registry()["corpora"] if c["name"] == name)
    if spec["type"] != "huggingface":
        raise NotImplementedError(f"Only HF parallel corpora are wired right now: {spec['type']}")
    from datasets import load_dataset  # type: ignore

    kwargs = {"split": spec.get("split", "train"), "streaming": True}
    if "config" in spec:
        ds = load_dataset(spec["repo"], spec["config"], **kwargs)
    else:
        ds = load_dataset(spec["repo"], **kwargs)
    en_field = spec["fields"]["en"]
    zh_field = spec["fields"]["zh"]
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        en = _resolve_field(item, en_field) or ""
        zh = _resolve_field(item, zh_field) or ""
        if en and zh:
            yield en.strip(), zh.strip(), name


def iter_text_pairs(
    rng: random.Random | None = None,
    max_total: int | None = None,
    enforce_length_check: bool = True,
) -> Iterator[tuple[str, str, str]]:
    """Round-robin iterator over text-only parallel corpora."""
    rng = rng or random.Random()
    registry = load_registry()["corpora"]
    weights = [c.get("sampling_weight", 1.0) for c in registry]
    iters = {c["name"]: iter_text_corpus(c["name"]) for c in registry}
    produced = 0
    while iters:
        name = rng.choices([c["name"] for c in registry], weights=weights, k=1)[0]
        if name not in iters:
            continue
        try:
            en, zh, src = next(iters[name])
        except StopIteration:
            iters.pop(name)
            continue
        if enforce_length_check:
            decision, _, _ = decide_paired_lengths(en, zh)
            if decision.drop_reason:
                continue
        yield en, zh, src
        produced += 1
        if max_total and produced >= max_total:
            return


def iter_image_caption_pairs(name: str, max_samples: int | None = None) -> Iterator[tuple[str, str, str, str]]:
    """Yield ``(image_url_or_path, en_caption, zh_caption, source)``."""
    spec = next(c for c in load_registry()["image_caption_corpora"] if c["name"] == name)
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(spec["repo"], split=spec.get("split", "train"), streaming=True)
    fimg, fen, fzh = spec["fields"]["image"], spec["fields"]["en"], spec["fields"]["zh"]
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        img = _resolve_field(item, fimg) or item.get(fimg)
        en = _resolve_field(item, fen) or ""
        zh = _resolve_field(item, fzh) or ""
        if img and en and zh:
            yield img, en.strip(), zh.strip(), name
