"""Paired length bin utilities for parallel zh/en sample construction.

Mirrors the paper's CleanTextSynth length bins {64, 128, 256, 512, 1024} on
the *English* side, while picking the natural Chinese length from the
parallel translation. We store both lengths in the alignment record rather
than truncating both sides to identical character counts (which would either
chop Chinese sentences or pad English with filler).

The empirical char ratio zh:en ≈ 0.55 (UN-PC / WMT22 / CCMatrix), so for
purely synthetic content we provide a "Chinese mirror bin" if a parallel
translation is unavailable. Token-level helpers use ``tiktoken`` if
installed; otherwise they fall back to character counts.
"""
from __future__ import annotations

from dataclasses import dataclass


# Paper's English-side bins.
EN_LENGTH_BINS_CHARS: tuple[int, ...] = (64, 128, 256, 512, 1024)

# Empirical Chinese mirror bins (approx. 0.55 × English).
ZH_MIRROR_BINS_CHARS: tuple[int, ...] = (36, 72, 144, 288, 576)

# Tolerance (in ratio) before we mark a pair as "length-imbalanced".
PAIR_LENGTH_TOLERANCE = 0.45


@dataclass
class PairedLengthDecision:
    bin_anchor: str          # e.g. "256"
    en_target_chars: int
    zh_target_chars: int
    en_actual_chars: int
    zh_actual_chars: int
    truncated_en: bool
    truncated_zh: bool
    drop_reason: str | None = None


def english_bin_for(length: int) -> int:
    for b in EN_LENGTH_BINS_CHARS:
        if length <= b:
            return b
    return EN_LENGTH_BINS_CHARS[-1]


def decide_paired_lengths(en_text: str, zh_text: str, en_target_bin: int | None = None) -> PairedLengthDecision:
    """Decide truncation/drop based on the paper's bins, working on raw character counts.

    Strategy:
      1. Pick the English bin: smallest bin >= len(en) (or the requested one).
      2. Truncate English at that bin (paper's behaviour for CleanTextSynth).
      3. Compute the natural Chinese bin index in :data:`ZH_MIRROR_BINS_CHARS`,
         and only truncate the zh side if it exceeds the mirror bin by > tol.
      4. If the absolute length ratio falls outside ``PAIR_LENGTH_TOLERANCE``
         (e.g. zh translation is suspiciously short), mark for drop.
    """
    en_target_bin = en_target_bin or english_bin_for(len(en_text))
    en_truncated = len(en_text) > en_target_bin
    en_kept = en_text[:en_target_bin]

    bin_idx = EN_LENGTH_BINS_CHARS.index(en_target_bin)
    zh_target = ZH_MIRROR_BINS_CHARS[bin_idx]
    # Allow up to 1.4x the mirror bin before truncating so we do not chop
    # natural sentences. If the zh text is truly long (e.g. WMT mismatch)
    # we still cap at 1.4x to keep on-image rendering manageable.
    zh_truncated = len(zh_text) > int(zh_target * 1.4)
    zh_kept = zh_text[: int(zh_target * 1.4)] if zh_truncated else zh_text

    drop = None
    if min(len(en_kept), 1) and min(len(zh_kept), 1):
        zh_per_en = len(zh_kept) / max(len(en_kept), 1)
        # Healthy range observed empirically: 0.30 .. 0.95.
        if zh_per_en < 0.30 or zh_per_en > 1.20:
            drop = f"length_ratio_out_of_range:{zh_per_en:.2f}"
    else:
        drop = "empty_side"

    return PairedLengthDecision(
        bin_anchor=str(en_target_bin),
        en_target_chars=en_target_bin,
        zh_target_chars=zh_target,
        en_actual_chars=len(en_kept),
        zh_actual_chars=len(zh_kept),
        truncated_en=en_truncated,
        truncated_zh=zh_truncated,
        drop_reason=drop,
    ), en_kept, zh_kept


def estimate_token_count(text: str, lang: str) -> int:
    """Rough token count. Falls back to char count if tiktoken is unavailable."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text)
