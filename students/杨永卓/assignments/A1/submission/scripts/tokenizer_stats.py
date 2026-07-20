#!/usr/bin/env python3
"""Compare tokenizer compression, longest token, and throughput."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cs336_basics.experiment import save_json
from cs336_basics.tokenizer import Tokenizer


def evaluate(tokenizer_path: str, text: str) -> dict[str, float | int | str]:
    tokenizer = Tokenizer.load(tokenizer_path)
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    longest = max(tokenizer.vocab.values(), key=len)
    byte_count = len(text.encode("utf-8"))
    return {
        "tokenizer": tokenizer_path,
        "sample_bytes": byte_count,
        "sample_tokens": len(ids),
        "compression_ratio_bytes_per_token": byte_count / len(ids),
        "throughput_bytes_per_sec": byte_count / elapsed,
        "longest_token_bytes": len(longest),
        "longest_token_utf8": longest.decode("utf-8", errors="replace"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", action="append", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.text, "rb") as source:
        sample = source.read(args.max_bytes).decode("utf-8", errors="ignore")
    save_json(args.output, {"results": [evaluate(path, sample) for path in args.tokenizer]})


if __name__ == "__main__":
    main()
