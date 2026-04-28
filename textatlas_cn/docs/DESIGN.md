# TextAtlas-CN 顶层方案设计

本文档对照原论文 *TextAtlas5M*，设计面向 **简体中文** 的复现框架。我们的目标是产出涵盖 9 个原子集 + 1 个评测集的中文富文本图像数据，所有子集共享统一的样本 schema、配置与质量管控流水线。

## 1. 总体架构

```
┌─────────────────────── 中文语料 / 主题种子 / 中文字体 / 公开图像源 ───────────────────────┐
│                                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  纯文本类     │  │  交错图文类   │  │  富场景合成   │  │  真实图像类   │                  │
│  │ CleanText    │  │ TextVision   │  │ StyledText   │  │ PPT/Paper/    │                  │
│  │ Synth-CN     │  │ Blend-CN     │  │ Synth-CN     │  │ Cover/Long/HQ │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                 │                          │
│         ▼                 ▼                 ▼                 ▼                          │
│ ┌──────────────────────────────────────────────────────────────────┐                     │
│ │ Sample Schema  (image, ocr_text[], bboxes[], scene_caption,       │                     │
│ │                font_attrs, prompt, meta, source_subset)           │                     │
│ └──────────────────────────────────────────────────────────────────┘                     │
│                                 │                                                        │
│                                 ▼                                                        │
│ ┌──────────────────────────────────────────────────────────────────┐                     │
│ │ Unified Caption Synthesizer  (Qwen-VL / GPT-4o + 600 中文模板)     │                     │
│ └──────────────────────────────────────────────────────────────────┘                     │
│                                 │                                                        │
│                                 ▼                                                        │
│ ┌──────────────────────────────────────────────────────────────────┐                     │
│ │  Quality Gate: OCR 校验 / NSFW / 水印 / 重复 / 语言检测 / 句长          │                     │
│ └──────────────────────────────────────────────────────────────────┘                     │
│                                 │                                                        │
│                                 ▼                                                        │
│                  WebDataset / JSONL / Parquet 导出                                       │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 中文语料来源

| 用途 | 推荐语料 | 备注 |
| --- | --- | --- |
| 纯长文本（CleanText、TextVision） | WuDaoCorpora 2.0、ChineseWebText-1.0、CCI3-HQ、Yi-Corpus、Skywork-Pile | 用于按 64/128/256/512/1024 字符截断 |
| 维基交错图文 | Chinese-Wiki + WIT-zh | 替代英文 WIT |
| 主题描述（StyledText/HQ） | 自建 18+26 主题种子；GPT-4o / Qwen2.5-72B 生成 | 见 `configs/topics_*.yaml` |
| 真实长文本场景 | LSVT、RCTW-17、ReCTS、MTWI、ICDAR2017-MLT、AnyWord-CN | 中文 OCR 评测集 |
| 论文/PPT | CNKI 开放、ChinaXiv、机器之心 PPT、第一 PPT、iSlide 公开 | |
| 封面 | 豆瓣读书公开 API + ISBN 数据库 | 代替 Amazon Cover Book |

## 3. 中文字体策略（约 200+）

`configs/fonts.yaml` 中将字体按风格分为 6 大类：黑体、宋体、仿宋、楷体、行书、艺术体；每条样本随机选择，并随机扰动：

- `font_size`: U(18, 72) 像素
- `font_color`: HSV 空间扰动，保证与背景对比度 > 30%
- `rotation`: U(-15°, +15°)
- `alignment`: {left, center, right, justify}
- `line_spacing`: U(1.0, 1.6)
- `字间距`: U(0.0, 0.15)*em

字体获取脚本 `textatlas_cn/scripts/prepare_fonts.py` 自动下载 Noto CJK SC、思源黑/宋、阿里巴巴普惠体、得意黑等开源字体并校验授权。

## 4. 模型选型

| 类别 | 推荐主选 | 备选 / 兜底 |
| --- | --- | --- |
| 文本 LLM | GPT-4o | Qwen2.5-72B-Instruct、DeepSeek-V3、GLM-4-Plus |
| 视觉 VLM | GPT-4o-vision | Qwen2.5-VL-72B、InternVL2.5-78B、MiniCPM-V-2.6 |
| 文生图 | CogView4-6B、Hunyuan-DiT 1.2、Kolors-1.5 | SD3.5 + Glyph-ByT5-CN |
| OCR | PaddleOCR PP-OCRv4-server-ch | EasyOCR ch_sim、CnOCR、Qwen2.5-VL OCR |
| 中文文本去重 | bge-base-zh-v1.5 + simhash | tencent-bert |
| 文本检测（场景填字） | YOLO11l (微调) | RT-DETR-r50vd、DBNet++ |
| 分割 | SAM2 | SAM-HQ |

LLM 调用统一封装于 `common/llm.py`，通过 `provider` 切换 OpenAI / DashScope / 智谱 / DeepSeek，并自动重试、限流、缓存（`diskcache`）。

## 5. 统一样本 Schema

```python
@dataclass
class TextAtlasSample:
    sample_id: str
    image_path: str              # 输出图像（PNG/JPG）
    width: int
    height: int
    source_subset: str           # 9 个子集名称之一
    language: str = "zh-Hans"
    ocr_lines: list[OcrLine]     # [{text, bbox(quad), font, size, color}]
    rendered_text: str           # 整图 OCR 文本（统一拼接）
    scene_caption: str           # VLM 描述（无文字）
    prompt: str                  # 训练用最终 prompt（合并 caption + OCR + 模板）
    layout_type: str             # "pure_text"|"interleaved"|"styled_scene"|...
    metadata: dict               # 字体、bbox 详细、分辨率、源 url、license
```

序列化为 JSONL，每行一个样本；图像独立保存于 `images/<shard>/<id>.png`，或打包为 webdataset shard。

## 6. 统一多模态描述（Unified Multimodal Data Construction）

对应论文 §3.4。我们维护 600 条中文模板（`data/templates/zh_caption_templates.txt`），按子集类型选择不同合并策略：

1. **pure_text**：`"一张白底图像，上面用 <font_style> 写着：'<rendered_text>'"`
2. **interleaved**：bullet 形式列出每个文本块和图像 caption，附带 bbox 与字体属性。
3. **styled_scene**：Qwen-VL 给出场景描述，并在文本区插入占位符 `<>`，再回填 `rendered_text`。
4. **real_dense_text**：调用 LLM 将 `scene_caption` 与 OCR 文本融合为自然段落（同论文 LongWordsSubset 处理方式）。

每个子集在 `subsets/<name>/captioning.py` 给出具体合并函数。

## 7. 各子集设计要点

### 7.1 CleanTextSynth-CN
- 输入：中文长文本（WuDao / ChineseWebText 抽样）。
- 处理：按 `length_bins=[64,128,256,512,1024]` 字符截断；用 PIL/OCR-Rendering 在 1024×1024 白底上渲染。
- 字体：随机从 200+ 中文字体；随机字号、颜色、对齐、旋转；多列布局随机。
- 输出：图像 + 完整文本 + 字体属性。无 VLM caption。

### 7.2 TextVisionBlend-CN
- 输入：Chinese-Wiki + WIT-zh 中带 1 张主图的章节，或 Obelics 中文页（Common Crawl 过滤 lang=zh）。
- 处理：PyMuPDF 生成 PDF；先随机放置 2–4 张图，再以最优空间填充文本；中文按行宽换行；每文本块 ≤ 50 字。
- 标注：解析 PDF 得到每个文本块/图像的 bbox、字体、字号；用 Qwen2.5-VL 给图像生中文 caption（≤ 50 字）。

### 7.3 StyledTextSynth-CN
- 主题种子：18 个高频中文场景（黑板教室、新闻播报、横幅、广告牌、电影海报、包装盒、说明书……）。
- 流程：
  1. GPT-4o 生成无文字场景 prompt（中文，≤ 160 字）。
  2. CogView4 / Kolors / SD3.5 生成 1024×1024 无字图。
  3. YOLO11/RT-DETR 检测可填字区，SAM2 细化为四边形。
  4. LLM 按场景生成中文文本（40–80 字），去重（bge + simhash）。
  5. 根据 bbox 形状选择矩形/不规则四边形渲染（PIL + 透视变换）。
- 拒绝采样规则：填字区过小、与其他主题相似、文字曲面、非真实感、bbox 不清、错文、低清晰度。

### 7.4 PPT2Details-CN
- 输入：iSlide / 第一PPT 的中文 PPTX。
- 流程：python-pptx 转 PDF（LibreOffice headless）→ PyMuPDF 转图；调用 Qwen2.5-VL，使用与论文一致 prompt 的中文版（见 §10）生成单段中文详细描述；过滤无字 / 低质量。

### 7.5 PPT2Structured-CN
- 输入：中文论文宣讲 PPTX 5k+（机器之心 talks、ChinaXiv 配套幻灯）。
- 流程：PyMuPDF 抽取每元素 bbox + 文本；图像元素用 Qwen-VL 生 caption；输出 JSON：`{elements:[{type,bbox,text|caption,font}], scene_caption}`。

### 7.6 Paper2Text-CN
- 输入：CNKI 开放/ChinaXiv 中文 PDF。
- 流程：PyMuPDF 按页解析；保留 font name/size/color；输出每页文本及字体属性；图像版本用 PyMuPDF 直接 render 成 PNG；prompt 由所有文本块字体+内容拼接。

### 7.7 CoverBook-CN
- 输入：豆瓣读书 / ISBN 数据库公开数据。
- 流程：抓取（仅元数据 + 公开封面 URL）→ 下载封面 → 拼接 caption：`{书名} | 作者：{author} | 出版：{publisher} {year} | 类别：{category}`；可附后封内容简介（≤ 300 字）。

### 7.8 LongWordsSubset-CN
- 输入：AnyWord-CN（AnyText 项目中文部分）、RCTW-17、LSVT、MTWI 2018、ReCTS。
- 过滤规则（中文化版）：
  1. **最少汉字数**：≥ 7 个汉字；
  2. **唯一字符比例** > 0.3；
  3. **连续重复**：禁止同一字连续 ≥ 3 次；
  4. **字符有效性**：剔除全是符号/数字的样本；
  5. **清洗**：去掉非中文/数字/标点的乱码；
  6. **空间排序**：bbox 按 top→bottom, left→right 排序。
- 输出：保留 caption + 排序后 OCR 文本。

### 7.9 TextScenesHQ-CN
- 输入：Common Crawl 中文 page、悟空 Wukong-100M、零一万物 Zero、Wechat 公众号封面合集（公开）。
- 流程：26 个中文主题（产品标签、广告牌、包装盒、显示屏、说明书、册页、手机截屏、墙贴、地贴、游戏直播、OLED、横幅、天气预报、公告板、新闻、黑板、数字屏、影院、指引牌、学术报告、校友档案、衣物文字、店招……）。
- 抓取 → PaddleOCR 过滤（≥ 10 词）→ Llama-3.1 / Qwen 拼写校正 → JSON 排序 → Qwen-VL 生背景描述（屏蔽图内文字）→ GPT-4o 用 500 个中文场景模板合成完整描述 → 人工复核（含人脸/水印/NSFW 检测）。

## 8. 评测集 TextAtlasEval-CN

- 1000 × CleanTextSynth-CN：从 ChineseWebText 中抽取 1000 段，按字符长度 64/128/256/512/1024 截断（每档 200 条）；
- 1000 × StyledTextSynth-CN：18 主题均匀采样；
- 1000 × TextScenesHQ-CN：26 主题均匀采样；
- 1000 × TextVisionBlend-CN：随机抽样；

所有样本人工校验：OCR 标注准确性、scene caption 与图像匹配、prompt 能否完整复现图像内容。

## 9. 质量保证

- **跨模型一致性**：StyledText / HQ 使用 Qwen-VL + InternVL 双模型生成 caption，输出不一致的进入人工复核；
- **重复检测**：bge-zh + simhash，Hamming 相似度 ≥ 0.9 视为重复；
- **NSFW / 水印 / 人脸**：NudeNet、watermark-detection-v0、RetinaFace-CN；
- **OCR 校验**：PaddleOCR + 大模型 OCR 双 source；CER ≤ 0.1 才入库；
- **语言纯度**：fastText lid-176 + jieba 中文比例 ≥ 0.95。

## 10. 中文 prompt 模板（节选）

`PPT2Details-CN`（对应论文同名 prompt 中文化）：

```
给定一张中文幻灯片图像，请将其中所有可见的视觉元素（包括文字段落、图表、表格、流程图等）整合为一段连贯、流畅、逻辑一致的中文描述。
要求：
1. 完整保留所有文字内容（含数字、专有名词、标点）；
2. 描述所有非文字视觉元素；
3. 不得遗漏或改写关键短语；
4. 仅输出一段话，不要分点。
```

`StyledTextSynth-CN` 场景生成（中文 General Prompt）：

```
我希望生成一张关于 <topic> 的图片，请输出一段不超过 160 字的中文场景描述。要求：
- 给出合理的 <topic> 描述；
- <topic> 正面朝向相机；
- <topic> 至少占据画面的三分之一；
- 包含一个复杂背景；
- <topic> 颜色统一，没有额外文字；
- 在画面中可见、清晰、未被其它物体遮挡。
```

`StyledTextSynth-CN` 文字生成：

```
请生成 50 段关于 <topic> 的中文文字，每段 30–60 字。
要求：可以是任意通知/标语/宣传词；只输出文字本体；保证文字之间含义不重复；输出格式 1.xxx 2.xxx ...
```

`Unified Caption (LongWordsSubset)`：

```
我有一段中文场景描述 T 和 OCR 文本 O，请生成 50 种自然组合方式，使描述与文字内容自然融合为更长的中文段落。
```

## 11. 风险与合规

1. **版权**：豆瓣封面、PPT、PDF 仅以 URL 索引方式分发；本仓库不再分发原图。
2. **隐私**：人脸检测后默认马赛克或剔除；
3. **生成式偏差**：GPT/Qwen 生成的中文文本通过敏感词表（中文常见敏感词 + 出版总署 NSFW 词表）过滤；
4. **API 成本**：所有 LLM/VLM 调用进入磁盘缓存（key=hash(prompt+model+params)）。
