from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal


BBox = tuple[float, float, float, float]
Polygon = list[tuple[float, float]]


@dataclass(slots=True)
class TextBlock:
    text: str
    bbox: BBox
    polygon: Polygon | None = None
    font: str | None = None
    font_size: float | None = None
    color: tuple[int, int, int] | None = None
    reading_order: int | None = None


@dataclass(slots=True)
class ImageBlock:
    path: str
    bbox: BBox
    caption: str | None = None
    reading_order: int | None = None


@dataclass(slots=True)
class DatasetSample:
    sample_id: str
    subset: str
    image_path: str
    prompt: str
    language: str = "zh"
    source: str | None = None
    split: Literal["train", "validation", "test"] = "train"
    text_blocks: list[TextBlock] = field(default_factory=list)
    image_blocks: list[ImageBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def write_jsonl(samples: Iterable[DatasetSample], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")
            count += 1
    return count


def append_jsonl(samples: Iterable[DatasetSample], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")
            count += 1
    return count


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
