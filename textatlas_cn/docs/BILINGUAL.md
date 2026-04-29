# TextAtlas-Parallel：完全平行的中英富文本图像数据

本文档描述如何在 `textatlas_cn` 框架内**同时**生成内容平行的中英两套富文本图像数据。两套图像在背景、布局、字体风格、bbox、`pair_id` 等维度严格对齐，差异**仅**在于渲染文字所用的语言（以及为该语言而选择的字体文件）。

## 1. 平行性两种模式

| 模式 | 子集 | 共享要素 | 差异要素 |
| --- | --- | --- | --- |
| `shared_layout`（**渲染端平行**） | CleanTextSynth、TextVisionBlend、StyledTextSynth | 画布尺寸、版式、bbox、底图（StyledTextSynth 的 T2I 底图严格相同）、字体 *风格*（zh↔en family 配对）、对齐/旋转/行距/颜色 | 渲染文字、字体 *文件*、最终图像 |
| `same_image`（**真实图像平行**） | PPT2Details、PPT2Structured、Paper2Text、CoverBook(可选)、LongWordsSubset、TextScenesHQ | **同一张图像**（不会重新渲染，因此源 OCR 文字也保留原状） | 双语 caption / prompt；OCR 文本的译文存于元数据 |

`ParallelTextAtlasSample.parallelism` 字段标记当前样本属于哪种模式。

## 2. 字符长度差异处理

中英在表达同一信息时字符数差异很大（zh:en ≈ 1:1.6 字符；token ≈ 1.6:1）。我们采用 4 项机制：

1. **以英文为锚的配对长度桶（paired length bins）**：保留论文 CleanTextSynth 的 5 个英文桶 `{64,128,256,512,1024}`；中文桶按经验 0.55× 设为 `{36,72,144,288,576}`，详见 `textatlas_cn/common/length_bins.py`。
2. **基于平行语料的真实译文**：CleanTextSynth-Parallel 不再"等字数截断"，而是从平行语料读取一对句子，按英文字符数选桶；中文版本使用对应译文，长度自然落在中文镜像桶内（允许 1.4× 弹性）。
3. **比例 sanity-check**：`decide_paired_lengths` 强制 0.30 ≤ `len(zh)/len(en)` ≤ 1.20，越界视作翻译错位丢弃。
4. **bbox 内自适应字号**：StyledText / TextScenesHQ 共享同一 bbox，渲染时由 `_fit_chinese_font` 二分搜索确定各自最大字号；最终中英字号可不同，但 bbox 与对齐方式完全一致。

## 3. 字体配对（zh ↔ en）

`textatlas_cn/common/font_pairs.py` 维护 6 对默认风格映射：

```
黑体  ↔ sans     (Noto Sans / Source Sans / Lato / Alibaba PuHuiTi)
宋体  ↔ serif    (Noto Serif / Source Serif Pro)
楷体  ↔ script   (Dancing Script / Great Vibes)
行书  ↔ script
行楷  ↔ script
艺术体 ↔ display  (Bebas Neue / Playfair Display)
```

每个 pair 共享 `shared_style` 标签（如 `modern_sans`），该标签会写入 `ParallelTextAtlasSample.shared.font_pair_style`，可用于训练时按风格分组。

下载：

```bash
python -m textatlas_cn.scripts.prepare_fonts
python -m textatlas_cn.scripts.prepare_fonts_en
```

## 4. 平行语料

详见 `configs/parallel_corpora.yaml`，主要包括：

- **通用平行语料**：UN-PC、CCMatrix、OPUS-100 (en-zh)、News Commentary v16、TED2020、AI Challenger 2017 Translation、WikiMatrix、WMT22。
- **图文 caption 平行**：Flickr30k-CN、COCO-CN、AIC-ICC、MUGE、WIT (zh+en)。
- **场景文本平行**：ICDAR2017/2019-MLT、MTWI-2018（自带中英双语标注）。

加载器在 `textatlas_cn/common/parallel_corpora.py`，提供 `iter_text_pairs()`（按 `sampling_weight` 混合）和 `iter_image_caption_pairs(name=...)`。

## 5. 翻译与质量校验

- **翻译器**：`textatlas_cn/common/translate.py` 默认使用 `LLMClient`（GPT-4o / Qwen2.5-72B / DeepSeek-V3 / GLM-4），Prompt 强制忠实翻译、保留实体/数字/标点。
- **跨语言相似度**：`bge-m3` 多语言 embedding 余弦相似度，阈值在 `configs/default.yaml` 中通过 `parallel.bge_m3_threshold` 配置（默认 0.78）。
- **回译 QA（可选）**：`back_translation_similarity()` 做 en→zh 回译并与原 zh 比较，阈值 0.55。
- **去重**：zh 用 `bge-base-zh-v1.5`，en 用 `bge-base-en-v1.5`；同一 pair 必须在两侧都不重复。

## 6. 各子集落地一览

| 子集 | Builder | 入口 | 关键平行策略 |
| --- | --- | --- | --- |
| CleanTextSynth-Parallel | `subsets/clean_text_synth/build_parallel.py` | `scripts/build_clean_text_synth_parallel.py` | 共享画布/版式参数 + 平行语料 + 配对字体 |
| TextVisionBlend-Parallel | `subsets/text_vision_blend/build_parallel.py` | `scripts/build_text_vision_blend_parallel.py` | 共用图像位置 + 双语 sections + 双 PDF |
| StyledTextSynth-Parallel | `subsets/styled_text_synth/build_parallel.py` | `scripts/build_styled_text_synth_parallel.py` | 共享 T2I 底图 + 共享 bbox + 双语文本对 |
| PPT2Details-Parallel | `subsets/ppt2details/build_parallel.py` | `scripts/build_ppt2details_parallel.py` | 同图 + Qwen-VL 双 prompt（论文原英文 prompt + 中文版） |
| PPT2Structured-Parallel | `subsets/ppt2structured/build_parallel.py` | `scripts/build_ppt2structured_parallel.py` | 同图 + 双语 element captions + 双语结构化 prompt |
| Paper2Text-Parallel | `subsets/paper2text/build_parallel.py` | `scripts/build_paper2text_parallel.py` | 同图 + 整页翻译保上下文 |
| CoverBook-Parallel | `subsets/cover_book/build_parallel.py` | `scripts/build_cover_book_parallel.py` | 同书的中英版本封面（OpenLibrary + 豆瓣 by ISBN）/同图翻译元信息 |
| LongWordsSubset-Parallel | `subsets/long_words_subset/build_parallel.py` | `scripts/build_long_words_subset_parallel.py` | 同图 + ICDAR-MLT/MTWI 自带双语 OCR 或翻译补齐 |
| TextScenesHQ-Parallel | `subsets/text_scenes_hq/build_parallel.py` | `scripts/build_text_scenes_hq_parallel.py` | 同图 + 双语 VLM 背景描述 + bge-m3 一致性校验 |
| TextAtlasEval-Parallel | `eval/build_eval_parallel.py` | `scripts/build_eval_parallel.py` | 在 `pair_id` 级别分层抽样，确保 zh/en 完全对齐 |

## 7. 输出格式

`ParallelJsonlWriter` 同时写出：

```
output/
├── parallel/<prefix>-XXXXX.jsonl   # 一行一个 ParallelTextAtlasSample
├── zh/<prefix>-XXXXX.jsonl         # 中文视图（含 pair_id）
├── en/<prefix>-XXXXX.jsonl         # 英文视图（含 pair_id）
├── images_zh/                      # shared_layout 子集才会有
├── images_en/
└── images_shared/                  # same_image 子集
```

最终统一打包脚本：

```bash
python -m textatlas_cn.scripts.unify_and_export_parallel \
    --inputs data/output/*/parallel/*.jsonl \
    --out data/output/textatlas_parallel_v1.jsonl --format jsonl
```

`webdataset` 模式会把 `<pair_id>.zh.<ext>`、`<pair_id>.en.<ext>` 与 `<pair_id>.json` 放进同一个 tar shard，方便流式训练。

## 8. 一键示例

```bash
# CleanTextSynth-Parallel：100k 对，自动按英文桶分层
python -m textatlas_cn.scripts.build_clean_text_synth_parallel \
    --num-pairs 100000 --output data/output/clean_text_synth_parallel

# StyledTextSynth-Parallel：18 主题 × 5k 对
python -m textatlas_cn.scripts.build_styled_text_synth_parallel \
    --topics configs/topics_styled.yaml --per-topic 5000 \
    --output data/output/styled_text_synth_parallel

# 评测集（4×1000 对）
python -m textatlas_cn.scripts.build_eval_parallel \
    --clean-text data/output/clean_text_synth_parallel/parallel/*.jsonl \
    --styled    data/output/styled_text_synth_parallel/parallel/*.jsonl \
    --scenes    data/output/text_scenes_hq_parallel/parallel/*.jsonl \
    --blend     data/output/text_vision_blend_parallel/parallel/*.jsonl
```
