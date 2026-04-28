"""Lightweight smoke tests that don't hit any external APIs.

We mainly verify imports, schema serialisation, OCR helpers, and the
template/caption merging utilities work end-to-end on hard-coded inputs.
"""
from __future__ import annotations

from textatlas_cn.common import schema, ocr, templates


def test_schema_roundtrip():
    sample = schema.TextAtlasSample(
        sample_id="abc",
        image_path="/tmp/x.png",
        width=10,
        height=20,
        source_subset="CleanTextSynth-CN",
        layout_type="pure_text",
        rendered_text="你好世界",
    )
    d = sample.to_dict()
    assert d["sample_id"] == "abc"
    assert d["language"] == "zh-Hans"


def test_chinese_ratio():
    assert ocr.chinese_ratio("你好abc") == 2 / 5
    assert ocr.has_consecutive_repeat("aaaab") is True
    assert ocr.has_consecutive_repeat("aaab") is False
    assert ocr.has_consecutive_repeat("abab") is False


def test_templates_render():
    out = templates.render_caption(
        templates.CaptionContext(scene_caption="一张红色的横幅", rendered_text="欢迎光临"),
        templates.build_default_templates(),
    )
    assert "欢迎光临" in out
