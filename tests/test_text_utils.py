from textatlas_zh_builder.text_utils import (
    TextFilterConfig,
    deduplicate_texts,
    is_valid_long_text,
    mixed_text_units,
    truncate_by_units,
)


def test_mixed_units_treat_chinese_chars_and_latin_words_as_units() -> None:
    assert mixed_text_units("欢迎来到 AI 2026 展会") == ["欢", "迎", "来", "到", "ai", "2026", "展", "会"]


def test_chinese_long_text_filter_rejects_repetition() -> None:
    config = TextFilterConfig(min_units=6, min_unique_ratio=0.3, max_consecutive_repeat=2)
    assert is_valid_long_text("新品发布会将展示智能家居方案", config)
    assert not is_valid_long_text("优惠 优惠 优惠 优惠", config)


def test_truncate_by_units_preserves_readable_prefix() -> None:
    assert truncate_by_units("智慧城市 AI 管理平台正式发布", 6) == "智慧城市 AI"


def test_lexical_deduplication_removes_near_identical_text() -> None:
    texts = ["新品发布会今日举行", "新品发布会今日举行", "城市交通实时更新"]
    assert deduplicate_texts(texts, ngram_size=2) == ["新品发布会今日举行", "城市交通实时更新"]
