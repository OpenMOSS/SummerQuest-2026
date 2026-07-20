#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import save_tokenizer, train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    elapsed = time.perf_counter() - start
    save_tokenizer(vocab, merges, args.output_prefix)
    metadata = {
        "input": Path(args.input).name,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_token,
        "training_time_sec": elapsed,
    }
    Path(f"{args.output_prefix}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
