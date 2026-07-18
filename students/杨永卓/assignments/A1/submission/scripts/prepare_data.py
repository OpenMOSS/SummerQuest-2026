#!/usr/bin/env python3
"""Resumable end-to-end data and tokenizer preparation pipeline."""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path


def run_if_missing(outputs: list[str], command: list[str]) -> None:
    if all(Path(path).exists() for path in outputs):
        print(f"skip existing: {', '.join(outputs)}", flush=True)
        return
    print("run:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    python = sys.executable
    Path("state").mkdir(exist_ok=True)
    Path("logs/tokenizers").mkdir(parents=True, exist_ok=True)
    Path("state/prepare.failed").unlink(missing_ok=True)
    try:
        run_if_missing(
            ["data/tinystories_train.txt", "data/tinystories_valid.txt", "data/owt_train.txt", "data/owt_valid.txt"],
            [python, "scripts/download_data.py", "--data-dir", "data"],
        )
        run_if_missing(
            ["data/tinystories_tokenizer.json", "logs/tokenizers/tinystories_train.json"],
            [
                python,
                "scripts/train_tokenizer.py",
                "--input",
                "data/tinystories_train.txt",
                "--vocab-size",
                "10000",
                "--output",
                "data/tinystories_tokenizer.json",
                "--summary",
                "logs/tokenizers/tinystories_train.json",
            ],
        )
        for split in ("train", "valid"):
            run_if_missing(
                [f"data/tinystories_{split}.bin", f"logs/tokenizers/tinystories_{split}_encode.json"],
                [
                    python,
                    "scripts/encode_data.py",
                    "--input",
                    f"data/tinystories_{split}.txt",
                    "--tokenizer",
                    "data/tinystories_tokenizer.json",
                    "--output",
                    f"data/tinystories_{split}.bin",
                    "--summary",
                    f"logs/tokenizers/tinystories_{split}_encode.json",
                ],
            )
        run_if_missing(
            ["data/owt_tokenizer.json", "logs/tokenizers/owt_train.json"],
            [
                python,
                "scripts/train_tokenizer.py",
                "--input",
                "data/owt_train.txt",
                "--vocab-size",
                "32000",
                "--output",
                "data/owt_tokenizer.json",
                "--summary",
                "logs/tokenizers/owt_train.json",
                "--max-input-bytes",
                "536870912",
            ],
        )
        for split in ("train", "valid"):
            run_if_missing(
                [f"data/owt_{split}.bin", f"logs/tokenizers/owt_{split}_encode.json"],
                [
                    python,
                    "scripts/encode_data.py",
                    "--input",
                    f"data/owt_{split}.txt",
                    "--tokenizer",
                    "data/owt_tokenizer.json",
                    "--output",
                    f"data/owt_{split}.bin",
                    "--summary",
                    f"logs/tokenizers/owt_{split}_encode.json",
                ],
            )
        for corpus in ("tinystories", "owt"):
            run_if_missing(
                [f"logs/tokenizers/{corpus}_comparison.json"],
                [
                    python,
                    "scripts/tokenizer_stats.py",
                    "--tokenizer",
                    "data/tinystories_tokenizer.json",
                    "--tokenizer",
                    "data/owt_tokenizer.json",
                    "--text",
                    f"data/{corpus}_valid.txt",
                    "--output",
                    f"logs/tokenizers/{corpus}_comparison.json",
                ],
            )
        Path("state/prepare.done").touch()
    except Exception:
        Path("state/prepare.failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
