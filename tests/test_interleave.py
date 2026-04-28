from pathlib import Path

from PIL import Image

from textatlas_zh_builder.interleave import InterleavedDocument, InterleavedItem, build_interleaved_sample


def test_build_interleaved_sample(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), "blue").save(image_path)
    document = InterleavedDocument(
        doc_id="doc1",
        items=[
            InterleavedItem(kind="title", text="中文标题"),
            InterleavedItem(kind="paragraph", text="这里是一段用于测试的中文段落内容。"),
            InterleavedItem(kind="image", image_path=str(image_path), caption="蓝色图片"),
        ],
    )

    sample = build_interleaved_sample(document, tmp_path / "out", seed=1)

    assert Path(sample.image_path).exists()
    assert len(sample.text_blocks) >= 2
    assert len(sample.image_blocks) == 1
    assert sample.prompt.startswith("生成一张中文图文混排页面")
