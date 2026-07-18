#!/usr/bin/env python3
"""Download the public TinyStories and OpenWebText assignment datasets."""

from __future__ import annotations

import argparse
import gzip
import shutil
import time
import urllib.request
from pathlib import Path


DATASETS = {
    "tinystories_train.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "tinystories_valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "owt_train.txt.gz": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
    "owt_valid.txt.gz": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
}


def download(url: str, destination: Path, max_retries: int = 12) -> None:
    if destination.exists():
        print(f"exists: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        downloaded = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={downloaded}-"})
        try:
            print(f"downloading: {url} from byte {downloaded} (attempt {attempt})", flush=True)
            with urllib.request.urlopen(request, timeout=120) as response:
                if downloaded and response.status != 206:
                    temporary.unlink(missing_ok=True)
                    downloaded = 0
                mode = "ab" if downloaded else "wb"
                with temporary.open(mode) as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                content_range = response.headers.get("Content-Range")
                expected_size = None
                if content_range and "/" in content_range:
                    expected_size = int(content_range.rsplit("/", 1)[1])
                elif response.headers.get("Content-Length"):
                    expected_size = downloaded + int(response.headers["Content-Length"])
            if expected_size is None or temporary.stat().st_size >= expected_size:
                temporary.replace(destination)
                return
        except Exception as error:
            print(f"download interrupted: {error}", flush=True)
        if attempt == max_retries:
            raise RuntimeError(f"download failed after {max_retries} attempts: {url}")
        time.sleep(min(5 * attempt, 30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DATASETS.items():
        destination = data_dir / filename
        download(url, destination)
        if destination.suffix == ".gz":
            extracted = destination.with_suffix("")
            if not extracted.exists():
                print(f"extracting: {destination}")
                with gzip.open(destination, "rb") as source, open(extracted, "wb") as target:
                    shutil.copyfileobj(source, target)


if __name__ == "__main__":
    main()
