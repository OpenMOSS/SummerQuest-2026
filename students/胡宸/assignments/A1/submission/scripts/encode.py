#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode UTF-8 text into uint16/uint32 token ids.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path, help="JSON mapping token id to hex bytes.")
    parser.add_argument("--merges", required=True, type=Path, help="Tab-separated hex-byte merge file.")
    parser.add_argument("--special-token", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    dtype = np.uint16 if len(tokenizer.vocab) < 2**16 else np.uint32
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.output.open("wb") as out:
        for line in src:
            ids = tokenizer.encode(line)
            if ids:
                np.asarray(ids, dtype=dtype).tofile(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
