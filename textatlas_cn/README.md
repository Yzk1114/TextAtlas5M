# TextAtlas-CN：中文长文本富图像数据集复现框架

本仓库面向论文 *TextAtlas5M: A Large-Scale Dataset for Long and Structured Text Image Generation* 中的 9 个子集（CleanTextSynth、TextVisionBlend、StyledTextSynth、PPT2Details、PPT2Structured、Paper2Text、CoverBook、LongWordsSubset、TextScenesHQ）以及评测集 TextAtlasEval，提供面向 **简体中文长文本/富文本图像数据** 构造的端到端复现管线。

> 整体方案见 [`docs/DESIGN.md`](docs/DESIGN.md)。各子集的具体实现位于 `textatlas_cn/subsets/<subset>/`。

## 与原论文关键差异

| 维度 | 原论文 (英文) | 本仓库 (中文) |
| --- | --- | --- |
| 文本语料 | Obelics、WIT、AnyWords3M、Marion10M | WuDao、Chinese-Wiki、悟空 (Wukong)、零一万物 Yi-Corpus、ChineseWebText、LSVT/RCTW/MTWI/ICDAR-ReCTS、AnyWord-CN |
| 字体 | 8,700 拉丁字体 | Noto Sans/Serif CJK SC、思源宋/黑、方正仿宋、楷体、华文系列、阿里巴巴普惠体、得意黑等约 200+ 中文字体 |
| 文生图模型 | SD3.5、PixArt-α | CogView4、Hunyuan-DiT 1.2、Kolors、Stable Diffusion 3.5 + Glyph-ByT5-CN（适配中文） |
| LLM | GPT-4o、Llama-3.1 | GPT-4o、Qwen2.5-72B、DeepSeek-V3、智谱 GLM-4 |
| VLM | Qwen2-VL、InternVL2、BLIP | Qwen2.5-VL、InternVL2.5、MiniCPM-V 2.6、GPT-4o-vision |
| OCR | EasyOCR (en) | **PaddleOCR (PP-OCRv4-ch)** + EasyOCR (ch_sim) + CnOCR + 大模型 OCR 校验 |
| PPT | SlideShare1M、AutoSlideGen | iSlide、第一PPT、CSDN/百度文库公开 PPT、中文论文宣讲幻灯 |
| 论文 | arXiv 英文 | CNKI 开放、PMLR 中文专题、中文期刊 PDF（公开数据） |
| 图书 | Cover Book (Amazon) | 豆瓣/当当公开图书封面（含 ISBN 检索） |
| 场景 | Common Crawl / LAION-5B (en) | 悟空、Common Crawl 中文 page、Zero (零一万物)、阿里 IC15-CN/M6-Chinese-Corpus |

## 目录结构

```
textatlas_cn/
├── README.md
├── docs/
│   └── DESIGN.md                # 顶层方案设计
├── configs/                     # YAML 配置（API key 通过 env 注入）
│   ├── default.yaml
│   ├── fonts.yaml
│   ├── topics_styled.yaml       # StyledTextSynth-CN 18 个主题
│   ├── topics_textscenes.yaml   # TextScenesHQ-CN 26 个主题
│   └── corpora.yaml
├── textatlas_cn/                # Python 包
│   ├── common/                  # 公共：schema、配置、API 封装、字体、OCR、IO
│   ├── subsets/                 # 每个子集一个子模块
│   │   ├── clean_text_synth/
│   │   ├── text_vision_blend/
│   │   ├── styled_text_synth/
│   │   ├── ppt2details/
│   │   ├── ppt2structured/
│   │   ├── paper2text/
│   │   ├── cover_book/
│   │   ├── long_words_subset/
│   │   └── text_scenes_hq/
│   ├── eval/                    # TextAtlasEval-CN 构造
│   ├── export/                  # 统一打包 (jsonl / webdataset / parquet)
│   └── scripts/                 # CLI 入口
├── data/                        # 默认数据/输出根目录（git 忽略）
└── tests/
```

## 安装

```bash
pip install -r requirements.txt
# 中文 OCR 与字体
pip install paddleocr cn_ocr
# 字体准备
python -m textatlas_cn.scripts.prepare_fonts
```

## 一键流程示例

```bash
# 1) 构造 CleanTextSynth-CN（200k 个，长度 [64,128,256,512,1024]）
python -m textatlas_cn.scripts.build_clean_text_synth \
    --num-samples 200000 --output data/output/clean_text_synth_cn

# 2) 构造 StyledTextSynth-CN（含 18 主题，每主题 5k）
python -m textatlas_cn.scripts.build_styled_text_synth \
    --topics configs/topics_styled.yaml --per-topic 5000

# 3) 合并打包并生成统一多模态描述
python -m textatlas_cn.scripts.unify_and_export \
    --inputs data/output/* --out data/output/textatlas_cn_v1.jsonl
```

## 与论文的对应关系

| 子集 | 论文规模 | 中文目标 | 关键脚本 |
| --- | --- | --- | --- |
| CleanTextSynth-CN | 1.9M | 0.3–1M | `build_clean_text_synth` |
| TextVisionBlend-CN | 547K | 100–300K | `build_text_vision_blend` |
| StyledTextSynth-CN | 426K | 100–300K | `build_styled_text_synth` |
| PPT2Details-CN | 298K | 50–200K | `build_ppt2details` |
| PPT2Structured-CN | 96K | 30–80K | `build_ppt2structured` |
| Paper2Text-CN | 356K | 50–200K | `build_paper2text` |
| CoverBook-CN | 207K | 50–150K | `build_cover_book` |
| LongWordsSubset-CN | 1.5M | 200–800K | `build_long_words_subset` |
| TextScenesHQ-CN | 36K | 10–30K | `build_text_scenes_hq` |
| TextAtlasEval-CN | 4K | 4K (4×1000) | `build_eval` |

## 安全与许可

- 所有外部数据需遵守源协议；本仓库仅提供数据构造管线代码，不分发原始数据。
- 调用第三方 API（OpenAI/Qwen/智谱/DeepSeek 等）需在 `.env` 中提供 key；不要把 key 写入仓库。
- 中文场景文本需谨慎处理人脸、品牌、版权信息；TextScenesHQ-CN 默认开启水印/NSFW/人脸检测过滤。
