# TextAtlas Chinese Dataset Builder

This repository contains reproduction code for the data construction pipeline
described in the paper.  The implementation focuses on building Chinese and
Chinese-English long-text rendering datasets while preserving the paper's split
design:

- **CleanTextSynthZH**: render long Chinese text on white canvases with random
  fonts, size, color, alignment, and small rotations.
- **TextVisionBlendZH**: synthesize interleaved text-image pages through
  PyMuPDF, render page images, and export parseable bounding-box annotations.
- **Paper2TextZH / PPT2StructuredZH**: extract page images and structured text
  blocks directly from PDF/PPT-converted PDF files.
- **LongWordsSubsetZH / TextScenesHQZH**: filter existing OCR/caption JSONL
  records with Chinese-aware length, repetition, and uniqueness rules.
- **Unified JSONL**: every subset is normalized into the same sample schema for
  training or evaluation.

## Installation

```bash
pip install -e .
```

Optional OCR and semantic dedup dependencies can be installed with:

```bash
pip install -e ".[ocr,semantic]"
```

## Quick start

Create a YAML config, or adapt `examples/chinese_pipeline.yaml`, then run:

```bash
textatlas-zh run examples/chinese_pipeline.yaml
```

Each configured step writes images/PDFs and a JSONL file under the selected
output directory.  The JSONL records use UTF-8 and keep Chinese characters
unescaped.

## Input formats

### Clean text

Plain UTF-8 text file.  Each non-empty line is one source document.

### Interleaved records

JSONL where each line contains:

```json
{
  "id": "doc-1",
  "segments": [
    {"type": "text", "text": "中文标题和段落"},
    {"type": "image", "path": "assets/example.jpg", "caption": "图片说明"}
  ]
}
```

### OCR/caption records

JSONL where each line contains an image path and OCR annotations:

```json
{
  "id": "sample-1",
  "image": "images/sample.jpg",
  "caption": "场景描述",
  "ocr": [
    {"text": "欢迎来到中文长文本数据集", "bbox": [10, 20, 300, 60]}
  ]
}
```

The filter accepts common aliases such as `image_path`, `text_blocks`,
`annotations`, `polygon`, and `points`.

## Unified sample schema

Each output JSONL line contains:

- `sample_id`
- `subset`
- `image_path`
- `prompt`
- `language`
- `split`
- `text_blocks`: text, bbox, font, color, reading order
- `image_blocks`: image path, bbox, optional caption
- `metadata`

## Why Chinese-specific code is needed

The paper's real-data filtering uses word count and word uniqueness.  Chinese
does not require whitespace between words, so this implementation counts CJK
characters as text units and keeps Latin/number spans as single units.  The same
unitization is used for truncation, long-text filtering, and lightweight lexical
deduplication.
