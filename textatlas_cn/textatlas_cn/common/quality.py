"""Quality-control utilities: language detection, dedup, NSFW/watermark filters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .ocr import chinese_ratio, unique_char_ratio, has_consecutive_repeat


@dataclass
class TextQualityConfig:
    min_chinese_ratio: float = 0.95
    min_unique_word_ratio: float = 0.30
    max_consecutive_repeat: int = 3
    min_text_chars: int = 7
    forbid_pattern_chars: str = ""


def passes_text_quality(text: str, cfg: TextQualityConfig) -> bool:
    if not text or len(text) < cfg.min_text_chars:
        return False
    if chinese_ratio(text) < cfg.min_chinese_ratio:
        return False
    if unique_char_ratio(text) < cfg.min_unique_word_ratio:
        return False
    if has_consecutive_repeat(text, cfg.max_consecutive_repeat):
        return False
    return True


# ----------------------------------------------------------------------
# Semantic deduplication using sentence-transformers + simhash.
# ----------------------------------------------------------------------
class SemanticDeduper:
    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5", threshold: float = 0.9, hash_bits: int = 64) -> None:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.hash_bits = hash_bits
        self._np = np
        self._proj = None
        self._hashes: list[int] = []

    def _hash(self, text: str) -> int:
        emb = self.model.encode([text], normalize_embeddings=True)[0]
        if self._proj is None:
            rng = self._np.random.RandomState(20260428)
            self._proj = rng.randn(self.hash_bits, emb.shape[0]).astype("float32")
        bits = self._proj @ emb
        out = 0
        for i, v in enumerate(bits):
            if v > 0:
                out |= 1 << i
        return out

    def is_duplicate(self, text: str) -> bool:
        h = self._hash(text)
        for prev in self._hashes:
            xor = bin(h ^ prev).count("1")
            sim = 1 - xor / self.hash_bits
            if sim >= self.threshold:
                return True
        self._hashes.append(h)
        return False


# ----------------------------------------------------------------------
# Image quality stubs (face/NSFW/watermark).
# ----------------------------------------------------------------------
class ImageQualityFilter:
    """Lightweight composite filter; backends are loaded lazily."""

    def __init__(self, ban_nsfw: bool = True, ban_watermark: bool = True, face_action: str = "blur") -> None:
        self.ban_nsfw = ban_nsfw
        self.ban_watermark = ban_watermark
        self.face_action = face_action  # "blur"|"drop"|"keep"
        self._nsfw = None
        self._watermark = None
        self._face = None

    def _ensure_nsfw(self):
        if self._nsfw is None:
            from nudenet import NudeDetector  # type: ignore
            self._nsfw = NudeDetector()
        return self._nsfw

    def check(self, image_path: str | Path) -> dict[str, Any]:
        result = {"nsfw": False, "watermark": False, "face_count": 0, "should_drop": False}
        if self.ban_nsfw:
            try:
                detector = self._ensure_nsfw()
                preds = detector.detect(str(image_path))
                bad = any(p.get("class") in {"FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED"} for p in preds)
                result["nsfw"] = bad
                if bad:
                    result["should_drop"] = True
            except Exception:
                pass
        # Watermark / face filters are placeholders; plug in your own models.
        return result
