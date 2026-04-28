"""Unified sample schema for all TextAtlas-CN subsets."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

LayoutType = Literal[
    "pure_text",
    "interleaved",
    "styled_scene",
    "ppt_details",
    "ppt_structured",
    "paper_text",
    "cover_book",
    "real_dense_text",
    "scene_hq",
]


@dataclass
class BBox:
    """Polygon bbox stored as 4 corner points (x, y) in image coordinates."""
    points: list[tuple[float, float]]
    label: str | None = None

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float, label: str | None = None) -> "BBox":
        return cls(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], label=label)


@dataclass
class FontAttrs:
    family: str
    size: float
    color: tuple[int, int, int]
    rotation: float = 0.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    alignment: str = "left"


@dataclass
class OcrLine:
    text: str
    bbox: BBox
    font: FontAttrs | None = None
    confidence: float | None = None


@dataclass
class TextAtlasSample:
    sample_id: str
    image_path: str
    width: int
    height: int
    source_subset: str
    layout_type: LayoutType
    rendered_text: str = ""                 # full image OCR / GT text joined with \n
    scene_caption: str = ""                 # background-only caption (no rendered text)
    prompt: str = ""                        # final unified caption used for training
    ocr_lines: list[OcrLine] = field(default_factory=list)
    language: str = "zh-Hans"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict converts BBox/FontAttrs but preserves tuples; ensure JSON-friendly.
        return d
