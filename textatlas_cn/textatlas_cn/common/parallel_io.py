"""I/O for parallel zh/en samples.

Writes three artefacts in lockstep:
- ``parallel/<prefix>-XXXXX.jsonl``: one row per pair (full ``ParallelTextAtlasSample``)
- ``zh/<prefix>-XXXXX.jsonl``: zh-only view (``TextAtlasSample`` rows)
- ``en/<prefix>-XXXXX.jsonl``: en-only view (``TextAtlasSample`` rows)

This lets downstream consumers either treat the dataset as bilingual pairs
or as two single-language datasets that can be cross-referenced via
``pair_id``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .parallel_schema import ParallelTextAtlasSample


class ParallelJsonlWriter:
    def __init__(self, out_dir: str | Path, prefix: str, shard_size: int = 5000) -> None:
        self.root = Path(out_dir)
        self.prefix = prefix
        self.shard_size = shard_size
        (self.root / "parallel").mkdir(parents=True, exist_ok=True)
        (self.root / "zh").mkdir(parents=True, exist_ok=True)
        (self.root / "en").mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self.count = 0
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        for attr in ("_pf", "_zf", "_ef"):
            fh = getattr(self, attr, None)
            if fh:
                fh.close()
        sx = f"{self.shard_idx:05d}"
        self._pf = (self.root / "parallel" / f"{self.prefix}-{sx}.jsonl").open("w", encoding="utf-8")
        self._zf = (self.root / "zh" / f"{self.prefix}-{sx}.jsonl").open("w", encoding="utf-8")
        self._ef = (self.root / "en" / f"{self.prefix}-{sx}.jsonl").open("w", encoding="utf-8")
        self.count = 0

    def write(self, pair: ParallelTextAtlasSample) -> None:
        d = pair.to_dict() if is_dataclass(pair) else pair
        # Both per-language rows are augmented with the pair id for joinability.
        zh_row = dict(d["zh"]); zh_row["pair_id"] = d["pair_id"]; zh_row["lang"] = "zh"
        en_row = dict(d["en"]); en_row["pair_id"] = d["pair_id"]; en_row["lang"] = "en"
        json.dump(d, self._pf, ensure_ascii=False); self._pf.write("\n")
        json.dump(zh_row, self._zf, ensure_ascii=False); self._zf.write("\n")
        json.dump(en_row, self._ef, ensure_ascii=False); self._ef.write("\n")
        self.count += 1
        if self.count >= self.shard_size:
            self.shard_idx += 1
            self._open_new_shard()

    def close(self) -> None:
        for attr in ("_pf", "_zf", "_ef"):
            fh = getattr(self, attr, None)
            if fh:
                fh.close()
                setattr(self, attr, None)

    def __enter__(self) -> "ParallelJsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
