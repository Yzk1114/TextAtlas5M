"""Download and validate the Chinese font registry declared in `configs/fonts.yaml`."""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

from ..common.config import load_config, resolve_path
from ..common.fonts import load_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    fonts_root = resolve_path(cfg, "fonts_root")

    registry = load_registry()
    for entry in tqdm(registry, desc="downloading fonts"):
        target: Path = fonts_root / entry["file"]
        if target.exists() and not args.force:
            continue
        url = entry["url"]
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        except Exception as exc:
            print(f"[warn] failed to fetch {entry['name']} from {url}: {exc}")


if __name__ == "__main__":
    main()
