from __future__ import annotations

import json

from textatlas_zh_builder.filtering import filter_long_words_jsonl
from textatlas_zh_builder.schema import load_jsonl
from textatlas_zh_builder.text_utils import TextFilterConfig


def test_filter_long_words_jsonl_keeps_chinese_long_text(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    source = tmp_path / "source.jsonl"
    output = tmp_path / "out.jsonl"
    rows = [
        {
            "image_path": str(image_path),
            "caption": "商场入口处的促销海报展示春季新品与会员优惠信息",
            "ocr": [
                {"text": "春季新品上市", "bbox": [0, 0, 100, 40]},
                {"text": "会员满三百减五十", "bbox": [0, 50, 150, 90]},
            ],
        },
        {
            "image_path": str(image_path),
            "caption": "短文本",
            "ocr": [{"text": "短", "bbox": [0, 0, 10, 10]}],
        },
    ]
    with source.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    count = filter_long_words_jsonl(
        source,
        output,
        config=TextFilterConfig(min_units=8, min_unique_ratio=0.2, min_cjk_ratio=0.5),
    )

    data = load_jsonl(output)
    assert count == 1
    assert len(data) == 1
    assert data[0]["subset"] == "LongWordsSubsetZH"
    assert "春季新品上市" in data[0]["prompt"]
