"""Build CleanTextSynth-CN.

Mirrors the paper's CleanTextSynth split:
- Sample long Chinese text from a mixed corpus (WuDao / ChineseWebText / Yi).
- Truncate to a length bin in {64, 128, 256, 512, 1024} characters.
- Render on a 1024x1024 white canvas using a randomly-sampled Chinese font
  with random size, color, alignment, line spacing and slight rotation.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.corpora import mixed_iter
from ...common.fonts import sample_font
from ...common.io import JsonlShardWriter, save_image, stable_id
from ...common.ocr import clean_text
from ...common.render import CleanTextRenderConfig, render_clean_text
from ...common.schema import OcrLine, TextAtlasSample


DEFAULT_LENGTH_BINS = (64, 128, 256, 512, 1024)


def _truncate_to_bin(text: str, target: int) -> str:
    text = clean_text(text)
    return text[:target] if len(text) > target else text


def build_clean_text_synth_cn(
    num_samples: int,
    output_dir: str | Path,
    length_bins: Iterable[int] = DEFAULT_LENGTH_BINS,
    config_path: str | Path | None = None,
    seed: int = 20260428,
) -> None:
    cfg = load_config(config_path)
    fonts_root = resolve_path(cfg, "fonts_root")
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    bins = list(length_bins)
    render_cfg = CleanTextRenderConfig()

    with JsonlShardWriter(meta_dir, "clean_text_synth_cn", shard_size=cfg["export"]["shard_size"]) as writer:
        produced = 0
        for raw_text, source in tqdm(mixed_iter(rng=rng, max_total=num_samples * 4), total=num_samples * 4):
            if produced >= num_samples:
                break
            target_len = rng.choice(bins)
            text = _truncate_to_bin(raw_text, target_len)
            if len(text) < min(target_len // 2, 32):
                continue

            font_entry = sample_font(fonts_root, rng=rng)
            try:
                img, font_attrs, bboxes = render_clean_text(
                    text=text, font_path=font_entry["path"], cfg=render_cfg, rng=rng
                )
            except Exception as exc:  # font missing glyphs etc.
                continue

            sid = stable_id(source, target_len, text[:64], font_entry["name"], rng.random())
            img_path = save_image(img, image_dir, sid, fmt=cfg["export"]["image_format"])
            sample = TextAtlasSample(
                sample_id=sid,
                image_path=img_path,
                width=img.size[0],
                height=img.size[1],
                source_subset="CleanTextSynth-CN",
                layout_type="pure_text",
                rendered_text=text,
                scene_caption="纯白背景，无场景内容。",
                prompt=f"白底图像，使用{font_entry['family']}字体写有：{text}",
                ocr_lines=[OcrLine(text=text, bbox=bboxes[0] if bboxes else None) for _ in [0]],
                metadata={
                    "source_corpus": source,
                    "length_bin": target_len,
                    "font": font_entry,
                    "font_attrs": font_attrs.__dict__,
                },
            )
            writer.write(sample.to_dict())
            produced += 1


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output", default="data/output/clean_text_synth_cn")
    parser.add_argument("--length-bins", nargs="*", type=int, default=list(DEFAULT_LENGTH_BINS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    build_clean_text_synth_cn(
        num_samples=args.num_samples,
        output_dir=args.output,
        length_bins=args.length_bins,
        config_path=args.config,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
