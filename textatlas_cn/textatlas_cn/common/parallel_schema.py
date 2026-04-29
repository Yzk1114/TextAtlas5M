"""Schema for the bilingual / parallel TextAtlas dataset.

Two parallelism modes are supported:

- ``shared_layout``: synthetic subsets (CleanTextSynth, TextVisionBlend,
  StyledTextSynth) where the *layout/base image is shared* and only the
  rendered text differs between zh and en.
- ``same_image``: real-image subsets (PPT2Details, PPT2Structured,
  Paper2Text, CoverBook, LongWordsSubset, TextScenesHQ) where the underlying
  image is *identical*; only the caption / prompt differs between zh and en.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from .schema import TextAtlasSample


ParallelismMode = Literal["shared_layout", "same_image"]


@dataclass
class AlignmentInfo:
    """Provenance of the zh/en text pairing."""
    source: str = ""              # e.g. "WMT22", "UN-PC", "Flickr30k-CN", "GPT-4o-translate"
    method: str = ""              # "human_pair" | "llm_translate" | "vlm_dual" | "metadata_pair"
    forward_model: str = ""       # model that produced en (or both)
    back_translation: str = ""    # optional: en -> zh -> en for QA
    bge_m3_sim: float | None = None
    laber_sim: float | None = None
    len_zh_chars: int = 0
    len_en_chars: int = 0
    len_zh_tokens: int = 0
    len_en_tokens: int = 0
    bin_anchor: str = ""          # the english length bin used (e.g. "256")


@dataclass
class ParallelTextAtlasSample:
    pair_id: str
    parallelism: ParallelismMode
    layout_type: str
    source_subset: str
    zh: TextAtlasSample
    en: TextAtlasSample
    shared: dict[str, Any] = field(default_factory=dict)   # bbox, t2i_prompt, base_image_hash, font_pair...
    alignment: AlignmentInfo = field(default_factory=AlignmentInfo)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
