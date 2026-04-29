"""Pack parallel zh/en samples into JSONL / WebDataset / Parquet.

The parallel variant emits **three** artefacts in a single tar/parquet/jsonl
file group: one row per pair plus per-language image members so consumers
can either iterate by pair or by language.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Iterable

from ..common.io import iter_jsonl


def pack_parallel(input_jsonls: Iterable[Path], output_path: Path, fmt: str = "jsonl") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with output_path.open("w", encoding="utf-8") as fh:
            for fp in input_jsonls:
                for sample in iter_jsonl(fp):
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return

    if fmt == "parquet":
        import pandas as pd
        rows: list[dict] = []
        for fp in input_jsonls:
            for sample in iter_jsonl(fp):
                rows.append(sample)
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        return

    if fmt == "webdataset":
        with tarfile.open(output_path, "w") as tar:
            for fp in input_jsonls:
                for sample in iter_jsonl(fp):
                    pid = sample["pair_id"]
                    for lang in ("zh", "en"):
                        sub = sample[lang]
                        img_path = Path(sub["image_path"])
                        if img_path.exists():
                            tar.add(img_path, arcname=f"{pid}.{lang}{img_path.suffix}")
                    info_bytes = json.dumps(sample, ensure_ascii=False).encode()
                    info = tarfile.TarInfo(f"{pid}.json")
                    info.size = len(info_bytes)
                    tar.addfile(info, io.BytesIO(info_bytes))
        return
    raise ValueError(f"Unsupported format: {fmt}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "parquet", "webdataset"])
    args = parser.parse_args()
    pack_parallel([Path(p) for p in args.inputs], Path(args.out), args.format)


if __name__ == "__main__":
    cli()
