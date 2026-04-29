from textatlas_zh_builder.rendering import RenderConfig, render_text_image


def test_render_text_image_creates_sample(tmp_path):
    image_path = tmp_path / "sample.png"
    sample = render_text_image(
        "这是一个用于测试的中文长文本数据样本，包含足够的信息量。",
        image_path,
        config=RenderConfig(width=320, height=240, margin=16, min_font_size=18, max_font_size=18),
    )

    assert image_path.exists()
    assert sample.language == "zh"
    assert sample.text_blocks[0].text.startswith("这是一个")
