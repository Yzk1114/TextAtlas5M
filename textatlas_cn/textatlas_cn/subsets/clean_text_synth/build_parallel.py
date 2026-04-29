"""Bilingual / parallel CleanTextSynth.

Renders each parallel zh/en text pair on **two** otherwise-identical canvases:
- the same canvas size, background colour, alignment, line spacing, rotation,
  and text colour;
- visually-paired font (zh family ↔ en family of the same typographic style);
- the English side picks the length bin (paper's 64/128/256/512/1024); the
  Chinese side keeps the natural translation length within the mirror bin.

Output uses :class:`ParallelJsonlWriter` to emit aligned ``parallel/``,
``zh/`` and ``en/`` shards plus per-language images at
``images_zh/`` and ``images_en/``.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from ...common.config import load_config, resolve_path
from ...common.font_pairs import sample_font_pair
from ...common.io import save_image, stable_id
from ...common.length_bins import EN_LENGTH_BINS_CHARS, decide_paired_lengths
from ...common.parallel_corpora import iter_text_pairs
from ...common.parallel_io import ParallelJsonlWriter
from ...common.parallel_schema import AlignmentInfo, ParallelTextAtlasSample
from ...common.render import CleanTextRenderConfig, render_clean_text
from ...common.render_en import CleanTextEnRenderConfig, render_clean_text_en
from ...common.schema import OcrLine, TextAtlasSample


def build_clean_text_synth_parallel(
    num_pairs: int,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    seed: int = 20260428,
    en_length_bins: tuple[int, ...] = EN_LENGTH_BINS_CHARS,
) -> None:
    cfg = load_config(config_path)
    fonts_root = resolve_path(cfg, "fonts_root")
    en_fonts_root = resolve_path(cfg, "fonts_en_root")
    img_zh = Path(output_dir) / "images_zh"
    img_en = Path(output_dir) / "images_en"
    img_zh.mkdir(parents=True, exist_ok=True)
    img_en.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    zh_cfg = CleanTextRenderConfig()
    en_cfg = CleanTextEnRenderConfig()

    with ParallelJsonlWriter(output_dir, "clean_text_synth_parallel", shard_size=cfg["export"]["shard_size"]) as writer:
        produced = 0
        # Iterate ~3x the requested pairs to allow length-filter rejections.
        for en_text, zh_text, src in tqdm(iter_text_pairs(rng=rng, max_total=num_pairs * 3), total=num_pairs * 3):
            if produced >= num_pairs:
                break
            target_bin = rng.choice(en_length_bins)
            decision, en_kept, zh_kept = decide_paired_lengths(en_text, zh_text, en_target_bin=target_bin)
            if decision.drop_reason:
                continue
            if len(en_kept) < min(target_bin // 2, 16) or len(zh_kept) < 8:
                continue

            try:
                zh_font, en_font = sample_font_pair(fonts_root, en_fonts_root, rng=rng)
            except FileNotFoundError:
                return

            # Sample shared rendering attributes ONCE so both languages match.
            font_size = rng.randint(*zh_cfg.font_size_range)
            rotation = rng.uniform(*zh_cfg.rotation_range)
            spacing = rng.uniform(*zh_cfg.line_spacing_range)
            alignment = rng.choice(zh_cfg.alignments)
            color = (rng.randint(0, 80),) * 3

            try:
                zh_img, zh_attrs, zh_bboxes = render_clean_text(
                    text=zh_kept, font_path=zh_font["path"], cfg=zh_cfg, rng=rng, text_color=color,
                )
                en_img, en_attrs, en_bboxes = render_clean_text_en(
                    text=en_kept,
                    font_path=en_font["path"],
                    cfg=en_cfg,
                    rng=rng,
                    text_color=color,
                    forced_font_size=font_size,
                    forced_alignment=alignment,
                    forced_rotation=rotation,
                    forced_line_spacing=spacing,
                )
            except Exception:
                continue

            pair_id = stable_id("CleanText", src, en_kept[:64], zh_kept[:32], target_bin)
            zh_path = save_image(zh_img, img_zh, pair_id, fmt=cfg["export"]["image_format"])
            en_path = save_image(en_img, img_en, pair_id, fmt=cfg["export"]["image_format"])

            zh_sample = TextAtlasSample(
                sample_id=f"{pair_id}-zh",
                image_path=zh_path,
                width=zh_img.size[0], height=zh_img.size[1],
                source_subset="CleanTextSynth-Parallel/zh",
                layout_type="pure_text",
                rendered_text=zh_kept,
                scene_caption="纯白背景，无场景内容。",
                prompt=f"白底图像，使用{zh_font['family']}字体写有：{zh_kept}",
                language="zh-Hans",
                ocr_lines=[OcrLine(text=zh_kept, bbox=zh_bboxes[0] if zh_bboxes else None)],
                metadata={
                    "source_corpus": src,
                    "font": zh_font, "font_attrs": zh_attrs.__dict__,
                    "length_bin": target_bin, "length_chars": decision.zh_actual_chars,
                },
            )
            en_sample = TextAtlasSample(
                sample_id=f"{pair_id}-en",
                image_path=en_path,
                width=en_img.size[0], height=en_img.size[1],
                source_subset="CleanTextSynth-Parallel/en",
                layout_type="pure_text",
                rendered_text=en_kept,
                scene_caption="A blank white background with no scene content.",
                prompt=f"A clean white image rendered in {en_font['name']} typeface, displaying: {en_kept}",
                language="en",
                ocr_lines=[OcrLine(text=en_kept, bbox=en_bboxes[0] if en_bboxes else None)],
                metadata={
                    "source_corpus": src,
                    "font": en_font, "font_attrs": en_attrs.__dict__,
                    "length_bin": target_bin, "length_chars": decision.en_actual_chars,
                },
            )
            pair = ParallelTextAtlasSample(
                pair_id=pair_id,
                parallelism="shared_layout",
                layout_type="pure_text",
                source_subset="CleanTextSynth-Parallel",
                zh=zh_sample,
                en=en_sample,
                shared={
                    "canvas_size": list(zh_cfg.canvas_size),
                    "alignment": alignment,
                    "rotation": rotation,
                    "line_spacing": spacing,
                    "text_color": list(color),
                    "font_pair_style": zh_font.get("shared_style"),
                },
                alignment=AlignmentInfo(
                    source=src,
                    method="human_pair",
                    forward_model="",
                    bge_m3_sim=None,
                    len_zh_chars=decision.zh_actual_chars,
                    len_en_chars=decision.en_actual_chars,
                    bin_anchor=str(target_bin),
                ),
                metadata={"truncated_zh": decision.truncated_zh, "truncated_en": decision.truncated_en},
            )
            writer.write(pair)
            produced += 1


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, required=True)
    parser.add_argument("--output", default="data/output/clean_text_synth_parallel")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    build_clean_text_synth_parallel(args.num_pairs, args.output, args.config, args.seed)


if __name__ == "__main__":
    cli()
