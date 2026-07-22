from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an exact number of BPE merges without saving a tokenizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--num-merges", type=positive_int, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--chunk-size-mb", type=positive_int, default=256)
    parser.add_argument("--pretoken-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = list(dict.fromkeys(args.special_token))
    initial_vocab_size = 256 + len(special_tokens)
    cache_preexisting = args.pretoken_cache is not None and args.pretoken_cache.exists()

    started = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=initial_vocab_size + args.num_merges,
        special_tokens=special_tokens,
        progress=args.progress,
        num_processes=args.workers,
        chunk_size_bytes=args.chunk_size_mb * 1024 * 1024,
        pretoken_cache_path=args.pretoken_cache,
    )
    elapsed = time.perf_counter() - started
    result = {
        "input_name": args.input.name,
        "requested_merges": args.num_merges,
        "completed_merges": len(merges),
        "vocab_size": len(vocab),
        "workers": args.workers,
        "chunk_size_mb": args.chunk_size_mb,
        "pretoken_cache_name": args.pretoken_cache.name if args.pretoken_cache else None,
        "cache_preexisting": cache_preexisting,
        "elapsed_sec": elapsed,
        "merges_per_sec_including_setup": len(merges) / elapsed if elapsed else None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
