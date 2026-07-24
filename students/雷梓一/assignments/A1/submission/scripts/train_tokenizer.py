from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer, train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-processes", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    vocab, merges = train_bpe(
        args.input,
        args.vocab_size,
        args.special_token,
        num_processes=args.num_processes,
    )
    tokenizer = Tokenizer(vocab, merges, args.special_token)
    tokenizer.save(args.output_dir / "vocab.json", args.output_dir / "merges.json")
    longest_token = b""
    for token in vocab.values():
        if len(token) > len(longest_token):
            longest_token = token
    summary = {
        "input": str(args.input),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_token,
        "num_processes": args.num_processes,
        "wall_clock_sec": time.perf_counter() - start,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "longest_token_bytes": len(longest_token),
        "longest_token_hex": longest_token.hex(),
        "longest_token_text": longest_token.decode("utf-8", errors="replace"),
    }
    (args.output_dir / "tokenizer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
