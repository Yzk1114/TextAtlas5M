from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", re.UNICODE)


@dataclass(frozen=True)
class TextFilterConfig:
    """Filtering thresholds adapted for Chinese and mixed Chinese-English text."""

    min_units: int = 10
    min_unique_ratio: float = 0.3
    max_consecutive_repeat: int = 3
    min_cjk_ratio: float = 0.0


def normalize_text(text: str, keep_punctuation: bool = True) -> str:
    """Normalize unicode width, whitespace, and optional punctuation noise."""

    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ")
    if not keep_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def cjk_chars(text: str) -> list[str]:
    return _CJK_RE.findall(text or "")


def mixed_text_units(text: str) -> list[str]:
    """Return Chinese characters plus Latin/number tokens as length units.

    The paper filters English-like long-word datasets by word count. Chinese text
    does not use whitespace consistently, so each CJK character is treated as a
    semantic unit while contiguous Latin letters and numbers remain word units.
    """

    units: list[str] = []
    token = []
    for char in normalize_text(text, keep_punctuation=False):
        if _CJK_RE.match(char):
            if token:
                units.append("".join(token).lower())
                token = []
            units.append(char)
        elif char.isalnum():
            token.append(char)
        else:
            if token:
                units.append("".join(token).lower())
                token = []
    if token:
        units.append("".join(token).lower())
    return units


def cjk_ratio(text: str) -> float:
    cleaned = normalize_text(text, keep_punctuation=False).replace(" ", "")
    if not cleaned:
        return 0.0
    return len(cjk_chars(cleaned)) / len(cleaned)


def has_excessive_consecutive_repetition(units: list[str], max_repeat: int) -> bool:
    if not units:
        return False
    current = units[0]
    run = 1
    for unit in units[1:]:
        if unit == current:
            run += 1
            if run > max_repeat:
                return True
        else:
            current = unit
            run = 1
    return False


def is_valid_long_text(text: str, config: TextFilterConfig | None = None) -> bool:
    config = config or TextFilterConfig()
    normalized = normalize_text(text, keep_punctuation=False)
    units = mixed_text_units(normalized)
    if len(units) < config.min_units:
        return False
    if config.min_cjk_ratio and cjk_ratio(normalized) < config.min_cjk_ratio:
        return False
    unique_ratio = len(set(units)) / max(1, len(units))
    if unique_ratio < config.min_unique_ratio:
        return False
    if has_excessive_consecutive_repetition(units, config.max_consecutive_repeat):
        return False
    return True


def truncate_by_units(text: str, max_units: int) -> str:
    """Truncate mixed Chinese-English text by CJK-char/Latin-token units.

    Whitespace-separated chunks are kept whole when possible, which avoids
    producing prompts such as ``AI 管`` from ``AI 管理平台``.
    """

    if max_units <= 0:
        return ""

    normalized = normalize_text(text)
    chunks = re.split(r"(\s+)", normalized)
    output: list[str] = []
    count = 0

    for chunk in chunks:
        if not chunk:
            continue
        if chunk.isspace():
            if output and not output[-1].isspace():
                output.append(" ")
            continue

        units = mixed_text_units(chunk)
        if not units:
            output.append(chunk)
            continue
        if count + len(units) <= max_units:
            output.append(chunk)
            count += len(units)
            continue
        if count == 0:
            partial: list[str] = []
            partial_count = 0
            latin_token: list[str] = []

            def flush_token() -> bool:
                nonlocal partial_count, latin_token
                if not latin_token:
                    return True
                if partial_count >= max_units:
                    latin_token = []
                    return False
                partial.extend(latin_token)
                latin_token = []
                partial_count += 1
                return partial_count < max_units

            for char in chunk:
                if _CJK_RE.match(char):
                    if not flush_token() or partial_count >= max_units:
                        break
                    partial.append(char)
                    partial_count += 1
                elif char.isalnum():
                    latin_token.append(char)
                    continue
                else:
                    if not flush_token() or partial_count >= max_units:
                        break
                    partial.append(char)
                if partial_count >= max_units:
                    break
            output.append("".join(partial))
        break
    return normalize_text("".join(output))


def stable_id(*parts: str, prefix: str = "") -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def deduplicate_texts(texts: list[str], ngram_size: int = 4, similarity_threshold: float = 0.9) -> list[str]:
    """Lightweight lexical deduplication for generated Chinese text.

    The paper uses sentence-transformer semantic hashing. This reproducible
    fallback avoids heavyweight model dependencies while keeping the same goal:
    remove near-identical text under the same topic.
    """

    kept: list[str] = []
    signatures: list[Counter[str]] = []
    for text in texts:
        units = mixed_text_units(text)
        if len(units) < ngram_size:
            grams = Counter(units)
        else:
            grams = Counter("".join(units[i : i + ngram_size]) for i in range(len(units) - ngram_size + 1))
        if not grams:
            continue
        duplicate = False
        for previous in signatures:
            intersection = sum((grams & previous).values())
            union = sum((grams | previous).values())
            if union and intersection / union >= similarity_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(text)
            signatures.append(grams)
    return kept
