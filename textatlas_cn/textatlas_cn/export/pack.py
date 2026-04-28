"""Pack assembled samples into JSONL / WebDataset / Parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ..common.io import iter_jsonl


def pack_dataset(input_jsonls: Iterable[Path], output_path: Path, fmt: str = "jsonl") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with output_path.open("w", encoding="utf-8") as fh:
            for fp in input_jsonls:
                for sample in iter_jsonl(fp):
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    elif fmt == "parquet":
        import pandas as pd
        rows: list[dict] = []
        for fp in input_jsonls:
            for sample in iter_jsonl(fp):
                rows.append(sample)
        pd.DataFrame(rows).to_parquet(output_path, index=False)
    elif fmt == "webdataset":
        import tarfile
        with tarfile.open(output_path, "w") as tar:
            for fp in input_jsonls:
                for sample in iter_jsonl(fp):
                    sid = sample["sample_id"]
                    img_path = Path(sample["image_path"])
                    if img_path.exists():
                        tar.add(img_path, arcname=f"{sid}.{img_path.suffix.lstrip('.')}")
                    info = json.dumps(sample, ensure_ascii=False).encode()
                    info_member = tarfile.TarInfo(f"{sid}.json")
                    info_member.size = len(info)
                    import io
                    tar.addfile(info_member, io.BytesIO(info))
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "parquet", "webdataset"])
    args = parser.parse_args()
    pack_dataset([Path(p) for p in args.inputs], Path(args.out), args.format)


if __name__ == "__main__":
    cli()
