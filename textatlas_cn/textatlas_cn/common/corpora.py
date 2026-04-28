"""Chinese corpus loaders.

Each corpus exposes :func:`iter_samples` returning text strings.
The default backend is HuggingFace `datasets` in streaming mode.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Iterator

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPORA_YAML = REPO_ROOT / "configs" / "corpora.yaml"


def load_registry(path: Path | None = None) -> list[dict]:
    p = Path(path or CORPORA_YAML)
    return yaml.safe_load(p.read_text(encoding="utf-8")).get("corpora", [])


def iter_corpus(name: str, max_samples: int | None = None) -> Iterator[str]:
    spec = next(c for c in load_registry() if c["name"] == name)
    if spec["type"] == "huggingface":
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(spec["repo"], split=spec.get("split", "train"), streaming=True)
        field = spec["field"]
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            text = item.get(field, "")
            if text:
                yield text
    elif spec["type"] == "local":
        path = Path(spec["path"])
        for p in sorted(path.glob("**/*.txt")):
            yield p.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported corpus type: {spec['type']}")


def mixed_iter(rng: random.Random | None = None, max_total: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield ``(text, source_corpus_name)`` according to sampling weights."""
    rng = rng or random.Random()
    registry = load_registry()
    weights = [c.get("sampling_weight", 1.0) for c in registry]
    iterators = {c["name"]: iter_corpus(c["name"]) for c in registry}
    total = 0
    while iterators:
        name = rng.choices([c["name"] for c in registry], weights=weights, k=1)[0]
        if name not in iterators:
            continue
        try:
            text = next(iterators[name])
            yield text, name
            total += 1
            if max_total and total >= max_total:
                return
        except StopIteration:
            iterators.pop(name)
