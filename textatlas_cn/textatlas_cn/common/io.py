"""I/O helpers: JSONL writer with sharding, image saving, hashing."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def stable_id(*parts: Any) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class JsonlShardWriter:
    """Round-robin JSONL writer producing files like prefix-000.jsonl."""

    def __init__(self, out_dir: str | Path, prefix: str, shard_size: int = 5000, gzip_output: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self.gzip_output = gzip_output
        self.shard_idx = 0
        self.count_in_shard = 0
        self._fh = None
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        if self._fh:
            self._fh.close()
        suffix = ".jsonl.gz" if self.gzip_output else ".jsonl"
        path = self.out_dir / f"{self.prefix}-{self.shard_idx:05d}{suffix}"
        self._fh = gzip.open(path, "wt", encoding="utf-8") if self.gzip_output else path.open("w", encoding="utf-8")
        self.count_in_shard = 0

    def write(self, sample: Any) -> None:
        if is_dataclass(sample):
            sample = asdict(sample)
        json.dump(sample, self._fh, ensure_ascii=False)
        self._fh.write("\n")
        self.count_in_shard += 1
        if self.count_in_shard >= self.shard_size:
            self.shard_idx += 1
            self._open_new_shard()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_image(img: Image.Image, out_dir: str | Path, sample_id: str, fmt: str = "png", quality: int = 95) -> str:
    out_dir = Path(out_dir)
    sub = out_dir / sample_id[:2]
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{sample_id}.{fmt.lower()}"
    if fmt.lower() in {"jpg", "jpeg"}:
        img.convert("RGB").save(path, quality=quality)
    else:
        img.save(path)
    return str(path)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
