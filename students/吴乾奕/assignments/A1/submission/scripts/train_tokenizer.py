#!/usr/bin/env python3
"""Train and serialize a byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 training corpus")
    parser.add_argument("--vocab-size", required=True, type=int, help="total vocabulary size")
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="special token; repeat this flag for multiple values",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=1 << 20, help="streaming text read size in characters")
    return parser.parse_args()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if __import__("sys").platform == "darwin" else value * 1024)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        chunk_size=args.chunk_size,
    )
    elapsed = time.perf_counter() - started

    tokenizer = Tokenizer(vocab, merges, args.special_token)
    paths = tokenizer.save(args.output_dir)
    longest = max(vocab.values(), key=len)
    report = {
        "input": str(args.input),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(vocab),
        "merge_count": len(merges),
        "special_tokens": args.special_token,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "longest_token_num_bytes": len(longest),
        "longest_token_hex": longest.hex(),
        "longest_token_utf8": longest.decode("utf-8", errors="replace"),
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    report_path = args.output_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, sort_keys=True)
        report_file.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
