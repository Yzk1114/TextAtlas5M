from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .text_utils import TextFilterConfig


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires installing PyYAML.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON/YAML object.")
    return data


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


@dataclass(frozen=True)
class RenderSettings:
    margin: int = 64
    min_font_size: int = 24
    max_font_size: int = 56


@dataclass(frozen=True)
class BuilderConfig:
    font_dirs: list[str] = field(default_factory=list)
    render: RenderSettings = field(default_factory=RenderSettings)
    text_filter: TextFilterConfig = field(default_factory=TextFilterConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuilderConfig":
        data = load_config(path)
        render_data = data.get("render", {})
        filter_data = data.get("text_filter", {})
        return cls(
            font_dirs=[str(item) for item in data.get("font_dirs", [])],
            render=RenderSettings(
                margin=int(render_data.get("margin", RenderSettings.margin)),
                min_font_size=int(render_data.get("min_font_size", RenderSettings.min_font_size)),
                max_font_size=int(render_data.get("max_font_size", RenderSettings.max_font_size)),
            ),
            text_filter=TextFilterConfig(
                min_units=int(filter_data.get("min_units", TextFilterConfig.min_units)),
                min_unique_ratio=float(filter_data.get("min_unique_ratio", TextFilterConfig.min_unique_ratio)),
                max_consecutive_repeat=int(
                    filter_data.get("max_consecutive_repeat", TextFilterConfig.max_consecutive_repeat)
                ),
                min_cjk_ratio=float(filter_data.get("min_cjk_ratio", TextFilterConfig.min_cjk_ratio)),
            ),
        )
