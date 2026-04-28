"""Caption template engine used to merge scene description and OCR text.

Reproduces the paper's "Unified Multimodal Data Construction" with 600 Chinese templates.
Templates can be loaded from `data/templates/zh_caption_templates.txt` (one per line)
or generated on-the-fly via :func:`build_default_templates`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_TEMPLATES_REL = "data/templates/zh_caption_templates.txt"


def build_default_templates() -> list[str]:
    """A small built-in template set; expand to 600 via LLM in production."""
    return [
        "在{scene_caption}的画面中，可以清晰地看到文字「{rendered_text}」。",
        "{scene_caption}画面里写着：{rendered_text}",
        "整张图描绘了{scene_caption}，其中显著呈现的文字内容为：{rendered_text}。",
        "{scene_caption}场景中包含以下文字信息——{rendered_text}",
        "图片以{scene_caption}为背景，文字部分为：「{rendered_text}」。",
        "本图展示了{scene_caption}，其上文字按从上到下从左到右的顺序为：{rendered_text}。",
        "请生成一张{scene_caption}的图像，画面中需要清楚地呈现文字：{rendered_text}。",
        "图像主体是{scene_caption}，其中渲染的中文文字内容为：{rendered_text}。",
        "{scene_caption}。文字内容（请按位置渲染）：{rendered_text}",
        "请绘制{scene_caption}，并在合适位置写入文字：{rendered_text}。",
    ]


def load_templates(path: str | Path | None = None) -> list[str]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / DEFAULT_TEMPLATES_REL
    p = Path(path)
    if p.exists():
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return build_default_templates()


@dataclass
class CaptionContext:
    scene_caption: str
    rendered_text: str
    layout_type: str = "styled_scene"
    bboxes: list | None = None
    fonts: list | None = None


def render_caption(ctx: CaptionContext, templates: Iterable[str], rng: random.Random | None = None) -> str:
    rng = rng or random
    tpl = rng.choice(list(templates))
    try:
        return tpl.format(scene_caption=ctx.scene_caption.strip(), rendered_text=ctx.rendered_text.strip())
    except (KeyError, IndexError):
        return f"{ctx.scene_caption} {ctx.rendered_text}"
