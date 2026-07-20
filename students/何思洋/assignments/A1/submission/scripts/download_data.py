from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


DATASETS = {
    "tinystories_train": (
        "TinyStoriesV2-GPT4-train.txt",
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    ),
    "tinystories_valid": (
        "TinyStoriesV2-GPT4-valid.txt",
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    ),
    "owt_train": (
        "owt_train.txt",
        "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
    ),
    "owt_valid": (
        "owt_valid.txt",
        "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
    ),
}


def download(url: str, output_path: Path, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"exists: {output_path}")
        return

    if url.endswith(".gz"):
        gz_path = output_path.with_suffix(output_path.suffix + ".gz")
        print(f"downloading: {url} -> {gz_path}")
        urllib.request.urlretrieve(url, gz_path)
        print(f"decompressing: {gz_path} -> {output_path}")
        with gzip.open(gz_path, "rb") as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz_path.unlink()
    else:
        print(f"downloading: {url} -> {output_path}")
        urllib.request.urlretrieve(url, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TinyStories and OWT sample data for A1.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for _, (filename, url) in selected.items():
        download(url, args.out_dir / filename, force=args.force)


if __name__ == "__main__":
    main()
