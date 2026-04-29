"""Download English fonts declared in `configs/fonts_en.yaml`."""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from ..common.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    fonts_root = Path(cfg["project_root"]) / cfg["paths"].get("fonts_en_root", "data/fonts_en")
    fonts_root.mkdir(parents=True, exist_ok=True)

    yaml_path = Path(cfg["project_root"]) / "configs" / "fonts_en.yaml"
    registry = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("fonts", [])
    for entry in tqdm(registry, desc="downloading en fonts"):
        target = fonts_root / entry["file"]
        if target.exists() and not args.force:
            continue
        try:
            resp = requests.get(entry["url"], timeout=120)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        except Exception as exc:
            print(f"[warn] failed to fetch {entry['name']}: {exc}")


if __name__ == "__main__":
    main()
