#!/usr/bin/env python3
"""Train and serialize a byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cs336_basics.experiment import save_json
from cs336_basics.tokenizer import Tokenizer, train_bpe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--max-input-bytes", type=int)
    args = parser.parse_args()
    start = time.perf_counter()
    training_stats: dict[str, int] = {}
    vocab, merges = train_bpe(
        args.input,
        args.vocab_size,
        args.special_token,
        progress=True,
        max_input_bytes=args.max_input_bytes,
        stats=training_stats,
    )
    elapsed = time.perf_counter() - start
    Tokenizer(vocab, merges, args.special_token).save(args.output)
    input_bytes = Path(args.input).stat().st_size
    save_json(
        args.summary,
        {
            "input_bytes": input_bytes,
            **training_stats,
            "sample_fraction": training_stats["sampled_input_bytes"] / input_bytes,
            "vocab_size": len(vocab),
            "merge_count": len(merges),
            "elapsed_sec": elapsed,
            "output": args.output,
        },
    )
    print(
        f"trained {len(vocab)} tokens from {training_stats['sampled_input_bytes']:,} bytes "
        f"in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
