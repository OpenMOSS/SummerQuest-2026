from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of pre-tokenization worker processes (default: up to 8)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token, num_processes=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": args.special_token}, file)
    print(f"trained {len(vocab)} tokens and {len(merges)} merges in {time.perf_counter() - start:.2f}s")
    print(args.output)


if __name__ == "__main__":
    main()
