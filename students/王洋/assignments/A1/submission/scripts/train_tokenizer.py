#!/usr/bin/env python3
"""Train and serialize a byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import cProfile
import json
import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe

from _common import PeakRSSMonitor, save_tokenizer_artifact, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train byte-level BPE and write tokenizer.json, vocab.json, and merges.txt.",
    )
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 training corpus")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty artifact directory")
    parser.add_argument("--vocab-size", type=int, required=True, help="total vocabulary size, including specials")
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="indivisible special token; repeat this option to add more than one",
    )
    parser.add_argument("--profile", type=Path, help="optional output path for cProfile statistics")
    parser.add_argument("--overwrite", action="store_true", help="replace tokenizer files in an existing directory")
    return parser.parse_args()


def validate_output(output_dir: Path, overwrite: bool) -> None:
    managed_files = ("tokenizer.json", "vocab.json", "merges.txt")
    existing = [name for name in managed_files if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(existing)
        raise FileExistsError(f"refusing to replace {joined} in {output_dir}; pass --overwrite to continue")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.vocab_size < 256 + len(set(args.special_token)):
        raise ValueError("vocab size must fit all 256 bytes and the requested special tokens")
    if len(args.special_token) != len(set(args.special_token)):
        raise ValueError("special tokens must not be repeated")
    validate_output(args.output_dir, args.overwrite)

    profiler = cProfile.Profile() if args.profile is not None else None
    start = time.perf_counter()
    with PeakRSSMonitor() as memory:
        if profiler is not None:
            profiler.enable()
        try:
            vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
        finally:
            if profiler is not None:
                profiler.disable()
    elapsed = time.perf_counter() - start

    if profiler is not None and args.profile is not None:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile)

    longest = sorted(vocab.items(), key=lambda item: (-len(item[1]), item[0]))[:20]
    summary = {
        "created_at_utc": utc_timestamp(),
        "source_filename": args.input.name,
        "source_bytes": args.input.stat().st_size,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": memory.peak_bytes,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "profile_filename": args.profile.name if args.profile is not None else None,
        "longest_tokens": [
            {
                "id": token_id,
                "length_bytes": len(token),
                "bytes_hex": token.hex(),
                "text": token.decode("utf-8", errors="backslashreplace"),
            }
            for token_id, token in longest
        ],
    }
    metadata = save_tokenizer_artifact(
        args.output_dir,
        vocab,
        merges,
        special_tokens=args.special_token,
        training_summary=summary,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
