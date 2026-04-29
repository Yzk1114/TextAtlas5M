"""Smoke tests for the parallel zh/en pipeline pieces that don't hit external APIs."""
from __future__ import annotations

import json
from pathlib import Path

from textatlas_cn.common.length_bins import (
    EN_LENGTH_BINS_CHARS,
    ZH_MIRROR_BINS_CHARS,
    decide_paired_lengths,
    english_bin_for,
)
from textatlas_cn.common.parallel_io import ParallelJsonlWriter
from textatlas_cn.common.parallel_schema import (
    AlignmentInfo,
    ParallelTextAtlasSample,
)
from textatlas_cn.common.schema import TextAtlasSample


def _make_sample(lang: str, text: str) -> TextAtlasSample:
    return TextAtlasSample(
        sample_id=f"x-{lang}",
        image_path=f"/tmp/x.{lang}.png",
        width=10, height=20,
        source_subset=f"Test/{lang}",
        layout_type="pure_text",
        rendered_text=text,
        language="zh-Hans" if lang == "zh" else "en",
    )


def test_english_bin_for():
    assert english_bin_for(40) == 64
    assert english_bin_for(64) == 64
    assert english_bin_for(65) == 128
    assert english_bin_for(2000) == 1024


def test_decide_paired_lengths_normal():
    en = "a" * 200
    zh = "甲" * 110  # zh:en ≈ 0.55, in healthy range
    decision, en_kept, zh_kept = decide_paired_lengths(en, zh, en_target_bin=256)
    assert decision.drop_reason is None
    assert decision.bin_anchor == "256"
    assert len(en_kept) == 200
    assert len(zh_kept) == 110


def test_decide_paired_lengths_truncate_en():
    en = "a" * 600
    zh = "甲" * 250
    decision, en_kept, zh_kept = decide_paired_lengths(en, zh, en_target_bin=512)
    assert decision.truncated_en is True
    assert len(en_kept) == 512
    assert decision.drop_reason is None


def test_decide_paired_lengths_drop_when_unbalanced():
    en = "this is a long english sentence " * 10
    zh = "短"  # absurdly short → drop
    decision, _, _ = decide_paired_lengths(en, zh, en_target_bin=128)
    assert decision.drop_reason is not None
    assert "length_ratio" in decision.drop_reason


def test_zh_mirror_bins_aligned():
    assert len(ZH_MIRROR_BINS_CHARS) == len(EN_LENGTH_BINS_CHARS)
    for zh, en in zip(ZH_MIRROR_BINS_CHARS, EN_LENGTH_BINS_CHARS):
        # mirror bin should be roughly half the english one
        assert 0.4 <= zh / en <= 0.7


def test_parallel_writer_emits_three_views(tmp_path: Path):
    pair = ParallelTextAtlasSample(
        pair_id="abc",
        parallelism="shared_layout",
        layout_type="pure_text",
        source_subset="Test",
        zh=_make_sample("zh", "你好世界"),
        en=_make_sample("en", "hello world"),
        shared={"font_pair_style": "modern_sans"},
        alignment=AlignmentInfo(source="unit-test", method="human_pair", len_zh_chars=4, len_en_chars=11, bin_anchor="64"),
    )
    with ParallelJsonlWriter(tmp_path, "test", shard_size=10) as w:
        w.write(pair)
    parallel_files = list((tmp_path / "parallel").glob("*.jsonl"))
    zh_files = list((tmp_path / "zh").glob("*.jsonl"))
    en_files = list((tmp_path / "en").glob("*.jsonl"))
    assert parallel_files and zh_files and en_files
    parallel_row = json.loads(parallel_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert parallel_row["pair_id"] == "abc"
    assert parallel_row["zh"]["rendered_text"] == "你好世界"
    assert parallel_row["en"]["rendered_text"] == "hello world"
    zh_row = json.loads(zh_files[0].read_text(encoding="utf-8").splitlines()[0])
    en_row = json.loads(en_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert zh_row["pair_id"] == "abc" and zh_row["lang"] == "zh"
    assert en_row["pair_id"] == "abc" and en_row["lang"] == "en"
