from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and serialize a byte-level BPE tokenizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--progress", action="store_true", help="Show BPE training progress on stderr.")
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=min(8, os.cpu_count() or 1),
        help="Worker processes for streaming pre-token counting (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=positive_int,
        default=256,
        help="Approximate pre-tokenization chunk size aligned to special-token boundaries (default: 256 MiB).",
    )
    parser.add_argument(
        "--pretoken-cache",
        type=Path,
        help="Optional trusted local pickle cache for the aggregated pre-token Counter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    vocab, merges = train_bpe(
        args.input,
        args.vocab_size,
        args.special_token,
        progress=args.progress,
        num_processes=args.workers,
        chunk_size_bytes=args.chunk_size_mb * 1024 * 1024,
        pretoken_cache_path=args.pretoken_cache,
    )
    elapsed = time.perf_counter() - started
    tokenizer = Tokenizer(vocab, merges, args.special_token)
    tokenizer.save(args.output)

    summary = {
        "input_name": args.input.name,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_token,
        "workers": args.workers,
        "chunk_size_mb": args.chunk_size_mb,
        "pretoken_cache_name": args.pretoken_cache.name if args.pretoken_cache else None,
        "elapsed_sec": elapsed,
        "output_name": args.output.name,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
