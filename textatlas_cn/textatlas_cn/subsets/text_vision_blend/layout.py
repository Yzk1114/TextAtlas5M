"""Page-layout planner for interleaved Chinese pages.

Strategy (mirrors paper §B.5.2):
- 2~4 images per page (Obelics-style) or 1 image (WIT-style).
- For WIT-style we keep a strict {title above image / main text below} structure
  to preserve semantic alignment with article sections.
- For Obelics-style we randomly position images, then fill remaining space with
  text boxes in left-top→right-bottom order, capping each block at 50 chars.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    type: str        # "image" | "text"
    role: str = ""   # "title" | "main" | "subtitle" | "caption" | ""
    payload: str | None = None  # text or image path


@dataclass
class PageLayout:
    page_w: float
    page_h: float
    boxes: list[Box]


def plan_obelics(images: list[str], texts: list[str], page_w: float = 1024, page_h: float = 1024,
                margin: float = 32, rng: random.Random | None = None) -> PageLayout:
    rng = rng or random.Random()
    boxes: list[Box] = []
    avail_y = margin
    cell_w = (page_w - 2 * margin)
    n_imgs = max(1, min(4, len(images)))
    rows = rng.choice([1, 2])
    cols = max(1, n_imgs // rows)
    img_h = (page_h * 0.45) / rows
    img_w = cell_w / cols
    for i, img in enumerate(images[: rows * cols]):
        r, c = i // cols, i % cols
        boxes.append(Box(
            x=margin + c * img_w + 4,
            y=avail_y + r * img_h + 4,
            w=img_w - 8,
            h=img_h - 8,
            type="image",
            payload=img,
        ))
    avail_y += rows * img_h + margin / 2

    text_box_h = max(80.0, (page_h - avail_y - margin) / max(len(texts), 1))
    for t in texts:
        if avail_y + text_box_h > page_h - margin:
            break
        boxes.append(Box(
            x=margin, y=avail_y, w=cell_w, h=text_box_h, type="text", payload=t[:200]
        ))
        avail_y += text_box_h + 6
    return PageLayout(page_w, page_h, boxes)


def plan_wit(image: str, sections: dict[str, str], page_w: float = 1024, page_h: float = 1024,
             margin: float = 48) -> PageLayout:
    boxes: list[Box] = []
    y = margin
    if title := sections.get("title"):
        boxes.append(Box(margin, y, page_w - 2 * margin, 60, type="text", role="title", payload=title))
        y += 70
    img_h = page_h * 0.4
    boxes.append(Box(margin, y, page_w - 2 * margin, img_h, type="image", role="figure", payload=image))
    y += img_h + 16
    if cap := sections.get("caption"):
        boxes.append(Box(margin, y, page_w - 2 * margin, 50, type="text", role="caption", payload=cap))
        y += 60
    if main := sections.get("main"):
        boxes.append(Box(margin, y, page_w - 2 * margin, page_h - y - margin, type="text", role="main", payload=main[:600]))
    return PageLayout(page_w, page_h, boxes)
