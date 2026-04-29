# TextAtlas 中文数据集构建工具

本仓库提供论文中数据构造流程的可复现代码，重点支持中文及中英混合长文本图像数据集构建。代码将论文中的多个数据分支统一为可运行的命令行流程，并输出一致的 UTF-8 JSONL 标注格式，便于后续训练、评测或人工质检。

## 支持的数据构造分支

- **CleanTextSynthZH**：在白色画布上渲染中文长文本，随机采样字体、字号、颜色、对齐方式和轻微旋转角度。
- **TextVisionBlendZH**：使用 PyMuPDF 生成可解析的图文交错 PDF 页面，同时导出页面 PNG 和文本/图片框标注。
- **Paper2TextZH / PPT2StructuredZH**：从 PDF 或由 PPT 转换得到的 PDF 中抽取页面图像、文本块、字体、字号、颜色和位置。
- **LongWordsSubsetZH / TextScenesHQZH**：对已有 OCR/字幕/场景描述 JSONL 进行中文长文本筛选、去重和阅读顺序排序。
- **统一 JSONL Schema**：所有分支最终写入同一种样本格式，方便混合训练或分层评测。

## 安装

建议使用 Python 3.9 及以上版本。

```bash
python3 -m pip install -e .
```

如需 OCR 或语义去重相关扩展依赖，可安装：

```bash
python3 -m pip install -e ".[ocr,semantic]"
```

如果安装后终端找不到 `textatlas-zh`，可以直接使用模块形式运行：

```bash
python3 -m textatlas_zh_builder.cli --help
```

## 字体准备

中文渲染需要可用的中文 TTF/OTF/TTC 字体。工具会默认扫描常见字体目录，例如：

- `/usr/share/fonts`
- `/usr/local/share/fonts`
- `~/.fonts`
- `~/.local/share/fonts`

也可以通过 `--font-dir` 显式指定字体目录：

```bash
textatlas-zh clean-text \
  --input data/texts.txt \
  --output-dir outputs/clean/images \
  --output-jsonl outputs/clean/annotations.jsonl \
  --font-dir /path/to/chinese/fonts
```

## 配置文件

`examples/chinese_pipeline.yaml` 提供了一份中文数据构建配置示例，包括随机种子、字体目录、渲染尺寸、文本过滤阈值和图文交错页面参数。各命令都可以通过 `--config` 读取该配置，并允许命令行参数覆盖关键选项。

```bash
textatlas-zh clean-text \
  --config examples/chinese_pipeline.yaml \
  --input data/texts.txt \
  --output-dir outputs/clean/images \
  --output-jsonl outputs/clean/annotations.jsonl
```

## 快速开始

### 1. 构建纯文本渲染数据

输入为 UTF-8 文本文件，每个非空行表示一条待渲染文本。

```text
智慧城市管理平台发布实时交通、公共安全、环境监测和社区服务信息，帮助居民快速了解城市运行状态。
```

运行：

```bash
textatlas-zh clean-text \
  --input data/texts.txt \
  --output-dir outputs/clean/images \
  --output-jsonl outputs/clean/annotations.jsonl \
  --limit 1000 \
  --width 1024 \
  --height 1024 \
  --max-units 256
```

输出：

- `outputs/clean/images/*.png`：渲染后的白底中文长文本图像
- `outputs/clean/annotations.jsonl`：统一格式标注

### 2. 构建图文交错页面数据

输入为 JSON 或 JSONL。每条记录可包含 `sections`，也兼容 `texts` / `text_segments` 与 `images` 的简化格式。

```json
{
  "id": "doc-1",
  "sections": [
    {"text": "中文标题和第一段正文。"},
    {"image": "assets/example.jpg", "caption": "配图说明"},
    {"text": "第二段正文描述图片中的关键信息。"}
  ]
}
```

运行：

```bash
textatlas-zh interleave \
  --input data/interleaved.jsonl \
  --output-dir outputs/interleave \
  --output-jsonl outputs/interleave/annotations.jsonl \
  --limit 1000
```

输出：

- `outputs/interleave/pdf/*.pdf`：可解析 PDF 页面
- `outputs/interleave/images/*.png`：渲染后的页面图像
- `outputs/interleave/annotations.jsonl`：文本框、图片框和生成提示词

### 3. 过滤已有 OCR/字幕数据

输入为 JSON 或 JSONL，每条记录包含图片路径、可选 caption，以及 OCR 文本块。

```json
{
  "id": "sample-1",
  "image_path": "images/sample.jpg",
  "caption": "商场入口处的促销海报",
  "ocr": [
    {"text": "春季新品上市", "bbox": [10, 20, 220, 60]},
    {"text": "会员满三百减五十", "bbox": [10, 80, 260, 120]}
  ]
}
```

运行：

```bash
textatlas-zh filter-ocr \
  --input data/ocr.jsonl \
  --output-jsonl outputs/long_words/annotations.jsonl \
  --subset LongWordsSubsetZH \
  --config examples/chinese_pipeline.yaml
```

该流程会执行：

1. 文本归一化；
2. 按中文字符与英文/数字 token 统计长度；
3. 过滤过短、重复度过高或连续重复的文本；
4. 按从上到下、从左到右排序 OCR 文本块；
5. 输出统一 JSONL 样本。

### 4. 从 PDF 构建文档页数据

输入目录包含 PDF 文件。PPT 数据可先转换为 PDF，再复用该流程。

```bash
textatlas-zh pdf \
  --input-dir data/pdfs \
  --output-dir outputs/pdf_pages \
  --output-jsonl outputs/pdf_pages/annotations.jsonl \
  --subset Paper2TextZH
```

输出包括渲染页图像、文本块 bbox、字体、字号、颜色，以及页面级 prompt。

## 统一输出格式

每一行 JSONL 对应一个样本，主要字段如下：

```json
{
  "sample_id": "clean_zh_xxx",
  "subset": "CleanTextSynthZH",
  "image_path": "outputs/clean/images/clean_zh_xxx.png",
  "prompt": "生成一张白色背景的中文长文本图片，图片中清晰排版以下文字：...",
  "language": "zh",
  "source": null,
  "split": "train",
  "text_blocks": [
    {
      "text": "中文文本",
      "bbox": [64.0, 120.0, 900.0, 180.0],
      "polygon": null,
      "font": "/usr/share/fonts/...",
      "font_size": 36.0,
      "color": [0, 0, 0],
      "reading_order": 0
    }
  ],
  "image_blocks": [],
  "metadata": {}
}
```

字段说明：

- `sample_id`：稳定样本 ID。
- `subset`：数据子集名称，如 `CleanTextSynthZH`。
- `image_path`：训练或评测使用的图像路径。
- `prompt`：用于文生图训练或评测的中文提示词。
- `text_blocks`：文本内容、位置、字体、字号、颜色和阅读顺序。
- `image_blocks`：图文交错页面中的图片路径、位置和可选 caption。
- `metadata`：源文件、页码、旋转角度、对齐方式等额外信息。

## 为什么需要中文专用处理

论文中的真实数据过滤流程主要基于英文“词数”和“唯一词比例”。中文文本通常没有空格分词，如果直接沿用英文词级规则，会错误过滤大量有效样本。因此本实现采用中文友好的混合单位：

- 每个 CJK 汉字计为一个文本单位；
- 连续英文或数字片段计为一个 token；
- 统一用于长度截断、长文本过滤、唯一比例统计和轻量近重复去重。

例如：

```python
mixed_text_units("欢迎来到 AI 2026 展会")
# ["欢", "迎", "来", "到", "ai", "2026", "展", "会"]
```

## 与论文流程的对应关系

| 论文数据分支 | 本代码命令 | 中文化实现重点 |
| --- | --- | --- |
| CleanTextSynth | `clean-text` | 中文字体扫描、中文换行、字符级长度控制 |
| TextVisionBlend | `interleave` | PyMuPDF 页面生成、图文框标注、中文 prompt |
| Paper2Text / PPT2Structured | `pdf` | PDF 页面渲染、文本 span 属性抽取 |
| LongWordsSubset / TextScenesHQ | `filter-ocr` | 中文长文本过滤、OCR 排序、去重 |

## 测试

运行单元测试：

```bash
python3 -m pytest
```

当前测试覆盖：

- 中文/中英混合文本单位统计；
- 中文长文本过滤；
- 文本截断与去重；
- CleanTextSynthZH 图像渲染；
- OCR JSONL 过滤；
- TextVisionBlendZH 图文交错样本生成。

## 常见问题

### 1. 渲染出的中文是方框或乱码

通常是缺少中文字体。请安装 Noto Sans CJK、思源黑体、文泉驿等字体，或通过 `--font-dir` 指向已有中文字体目录。

### 2. `textatlas-zh` 命令不可用

可能是用户级安装目录没有加入 `PATH`。可以使用：

```bash
python3 -m textatlas_zh_builder.cli --help
```

### 3. OCR 输入字段名不一致怎么办

过滤器兼容常见别名：

- 图片路径：`image_path`、`image`、`path`
- OCR 列表：`text_blocks`、`ocr`、`annotations`
- 文本字段：`text`、`transcription`、`label`
- 位置字段：`bbox`、`box`、`polygon`、`points`
