"""Prompt templates for StyledTextSynth-CN, mirroring the paper's Tables in Appendix.

Three families:
- ``general_scene_prompt`` : §B.1.1 General Prompt → CN
- ``seed_scene_prompt`` : §B.1.1 Topic with Human-Designed Seeds → CN
- ``text_for_topic_prompt`` : §B.1.2 Scene-dependent text → CN
- ``vlm_text_for_image_prompt`` : §B.1.2 Visual-Dependent Scenes → CN
"""
from __future__ import annotations


GENERAL_SCENE_PROMPT = (
    "我希望生成一张关于 {topic} 的中文图像，请输出一段不超过160字的中文场景描述，"
    "要求：\n"
    "- 给出合理的 {topic} 描述；\n"
    "- {topic} 正面朝向相机；\n"
    "- {topic} 至少占据画面的三分之一；\n"
    "- 包含一个复杂的背景；\n"
    "- {topic} 颜色统一，且不要包含任何文字；\n"
    "- 画面中可见、清晰、未被其它物体遮挡。\n"
    "只输出场景描述本身，不要分点。"
)


SEED_SCENE_PROMPT = (
    "请基于以下种子描述，扩写为一段不超过160字的中文图像生成 prompt，要求保留种子描述的核心结构，"
    "并补充镜头、光照、场景细节，画面中不要包含任何文字：\n"
    "种子描述：{seed}"
)


TEXT_FOR_TOPIC_PROMPT_GPT = (
    "请生成 {n} 段关于 {topic} 的中文文字，每段 {min_len}~{max_len} 字。\n"
    "要求：可以是任意通知、标语、宣传或说明文案；只输出文字本体；保证段落之间含义不重复；"
    "输出格式为：1.xxx\\n2.xxx\\n…"
)


TEXT_FOR_TOPIC_PROMPT_QWEN = TEXT_FOR_TOPIC_PROMPT_GPT  # text-only fallback


VLM_TEXT_FOR_IMAGE_PROMPT = (
    "这是一张关于 {topic} 的图像。请基于图像内容想象一个合适的中文场景文案，"
    "并按下列字段返回纯 JSON：\n"
    "{{\"title\": <≤6字标题>, \"body\": <30~60字正文>}}"
)
