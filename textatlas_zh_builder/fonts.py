from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/System/Library/Fonts",
    "/Library/Fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / ".local/share/fonts"),
)

_CJK_FONT_HINTS = (
    "noto",
    "sourcehan",
    "source-han",
    "思源",
    "wqy",
    "wenquanyi",
    "droid",
    "cjk",
    "song",
    "hei",
    "kai",
    "fang",
    "ming",
    "gothic",
    "simsun",
    "simhei",
    "msyh",
    "pingfang",
)


@dataclass(frozen=True)
class FontCatalog:
    fonts: tuple[Path, ...]

    @classmethod
    def from_directories(cls, font_dirs: list[str | Path] | tuple[str | Path, ...] | None = None) -> "FontCatalog":
        return discover_fonts(list(font_dirs) if font_dirs else None)

    def require(self) -> tuple[Path, ...]:
        if not self.fonts:
            searched = ", ".join(DEFAULT_FONT_DIRS)
            raise FileNotFoundError(
                "No TrueType/OpenType fonts were found. Install a Chinese font "
                f"or pass --font-dir explicitly. Searched: {searched}"
            )
        return self.fonts

    def random_font(self, rng) -> Path | None:
        if not self.fonts:
            return None
        return rng.choice(self.fonts)


def discover_fonts(font_dirs: list[str | Path] | None = None, prefer_cjk: bool = True) -> FontCatalog:
    dirs = [Path(p).expanduser() for p in (font_dirs or DEFAULT_FONT_DIRS)]
    seen: set[Path] = set()
    fonts: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for root, _, files in os.walk(directory):
            for filename in files:
                if not filename.lower().endswith((".ttf", ".ttc", ".otf")):
                    continue
                path = Path(root) / filename
                if path not in seen:
                    seen.add(path)
                    fonts.append(path)
    if prefer_cjk:
        fonts.sort(key=lambda p: (not _looks_like_cjk_font(p), str(p).lower()))
    else:
        fonts.sort(key=lambda p: str(p).lower())
    return FontCatalog(tuple(fonts))


def _looks_like_cjk_font(path: Path) -> bool:
    name = path.name.lower()
    return any(hint in name for hint in _CJK_FONT_HINTS)
