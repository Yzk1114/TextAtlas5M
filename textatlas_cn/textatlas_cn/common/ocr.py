"""OCR engine wrapper.

Primary: PaddleOCR PP-OCRv4 Chinese.
Backups: EasyOCR (ch_sim), CnOCR.
For high-stakes verification we additionally support Qwen2.5-VL OCR via :class:`LLMClient`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .schema import BBox, OcrLine


@dataclass
class OcrResult:
    lines: list[OcrLine]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(line.text) for line in self.lines)


class ChineseOCR:
    def __init__(self, primary: str = "paddleocr", **kwargs: Any) -> None:
        self.primary = primary
        self._engine: Any = None
        self._kwargs = kwargs

    def _ensure(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self.primary == "paddleocr":
            from paddleocr import PaddleOCR  # type: ignore
            self._engine = PaddleOCR(use_angle_cls=True, lang="ch", **self._kwargs)
        elif self.primary == "easyocr":
            import easyocr  # type: ignore
            self._engine = easyocr.Reader(["ch_sim", "en"], gpu=self._kwargs.get("use_gpu", False))
        elif self.primary == "cnocr":
            from cnocr import CnOcr  # type: ignore
            self._engine = CnOcr()
        else:
            raise ValueError(f"Unknown OCR engine: {self.primary}")
        return self._engine

    # ------------------------------------------------------------------
    def read(self, image: str | Path | Image.Image | np.ndarray) -> OcrResult:
        engine = self._ensure()
        if isinstance(image, (str, Path)):
            arr = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = image

        if self.primary == "paddleocr":
            return self._read_paddle(engine, arr)
        if self.primary == "easyocr":
            return self._read_easy(engine, arr)
        if self.primary == "cnocr":
            return self._read_cnocr(engine, arr)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    def _read_paddle(self, engine, arr) -> OcrResult:
        result = engine.ocr(arr, cls=True)
        lines: list[OcrLine] = []
        if not result:
            return OcrResult(lines)
        for page in result:
            if not page:
                continue
            for poly, (txt, conf) in page:
                bbox = BBox(points=[(float(x), float(y)) for x, y in poly])
                lines.append(OcrLine(text=txt, bbox=bbox, confidence=float(conf)))
        return OcrResult(lines)

    def _read_easy(self, engine, arr) -> OcrResult:
        result = engine.readtext(arr)
        lines: list[OcrLine] = []
        for poly, txt, conf in result:
            bbox = BBox(points=[(float(x), float(y)) for x, y in poly])
            lines.append(OcrLine(text=txt, bbox=bbox, confidence=float(conf)))
        return OcrResult(lines)

    def _read_cnocr(self, engine, arr) -> OcrResult:
        result = engine.ocr(arr)
        lines: list[OcrLine] = []
        for item in result:
            poly = item.get("position")
            txt = item.get("text", "")
            conf = float(item.get("score", 0.0))
            if poly is None:
                continue
            bbox = BBox(points=[(float(x), float(y)) for x, y in poly])
            lines.append(OcrLine(text=txt, bbox=bbox, confidence=conf))
        return OcrResult(lines)


# ----------------------------------------------------------------------
# Sorting / cleaning helpers (mirroring the paper's filtering pipeline)
# ----------------------------------------------------------------------
_NON_CJK_PUNC = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9，。、！？：；“”‘’（）《》〈〉【】「」〔〕\s\-\.,\!\?\:\;\(\)\[\]]")


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(len(text), 1)


def unique_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(set(text)) / max(len(text), 1)


def has_consecutive_repeat(text: str, max_run: int = 3) -> bool:
    run, prev = 1, None
    for ch in text:
        if ch == prev:
            run += 1
            if run > max_run:
                return True
        else:
            run, prev = 1, ch
    return False


def clean_text(text: str) -> str:
    text = _NON_CJK_PUNC.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sort_ocr_lines(lines: Sequence[OcrLine], y_tolerance: int = 12) -> list[OcrLine]:
    """Top→bottom, left→right ordering as in the paper's appendix."""
    def key(line: OcrLine) -> tuple[int, float]:
        ys = [p[1] for p in line.bbox.points]
        xs = [p[0] for p in line.bbox.points]
        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)
        # Bucketize y to handle near-equal rows.
        return (int(cy // y_tolerance), cx)
    return sorted(lines, key=key)
