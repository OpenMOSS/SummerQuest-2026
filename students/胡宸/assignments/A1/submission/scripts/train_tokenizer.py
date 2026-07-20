#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics.tokenizer import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--vocab-out", required=True, type=Path)
    parser.add_argument("--merges-out", required=True, type=Path)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    args.vocab_out.write_text(
        json.dumps({idx: token.hex() for idx, token in vocab.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.merges_out.write_text(
        "\n".join(f"{left.hex()}\t{right.hex()}" for left, right in merges) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
