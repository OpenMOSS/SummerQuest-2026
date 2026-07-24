#!/usr/bin/env python3
"""Measure tokenizer compression and streaming throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

from _common import PeakRSSMonitor, atomic_write_json, iter_text_chunks, load_tokenizer_artifact, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one tokenizer/corpus pairing and emit JSON statistics.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="tokenizer directory or tokenizer.json")
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 corpus")
    parser.add_argument("--chunk-chars", type=int, default=1024 * 1024)
    parser.add_argument("--max-chars", type=int, help="optional prefix length for quick or matched comparisons")
    parser.add_argument("--warmup-chars", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=3, help="number of measured full passes")
    parser.add_argument("--errors", choices=("strict", "replace", "ignore"), default="strict")
    parser.add_argument("--output", type=Path, help="optional JSON output file")
    return parser.parse_args()


def encode_once(tokenizer: object, args: argparse.Namespace, max_chars: int | None) -> tuple[int, int, int]:
    byte_count = 0
    char_count = 0

    def measured_chunks() -> Iterator[str]:
        nonlocal byte_count, char_count
        for chunk in iter_text_chunks(
            args.input,
            args.chunk_chars,
            max_chars=max_chars,
            errors=args.errors,
        ):
            char_count += len(chunk)
            byte_count += len(chunk.encode("utf-8"))
            yield chunk

    token_count = sum(1 for _ in tokenizer.encode_iterable(measured_chunks()))
    return token_count, byte_count, char_count


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.chunk_chars <= 0 or args.repeat <= 0:
        raise ValueError("chunk-chars and repeat must be positive")
    if args.max_chars is not None and args.max_chars <= 0:
        raise ValueError("max-chars must be positive")
    if args.warmup_chars < 0:
        raise ValueError("warmup-chars must be non-negative")

    tokenizer, metadata, metadata_path = load_tokenizer_artifact(args.tokenizer)
    if args.warmup_chars:
        encode_once(tokenizer, args, min(args.warmup_chars, args.max_chars or args.warmup_chars))

    durations: list[float] = []
    observed: tuple[int, int, int] | None = None
    with PeakRSSMonitor() as memory:
        for _ in range(args.repeat):
            start = time.perf_counter()
            current = encode_once(tokenizer, args, args.max_chars)
            durations.append(time.perf_counter() - start)
            if observed is not None and current != observed:
                raise RuntimeError("repeated tokenizer passes produced inconsistent counts")
            observed = current

    assert observed is not None
    token_count, byte_count, char_count = observed
    total_bytes = byte_count * args.repeat
    total_seconds = sum(durations)
    median_seconds = statistics.median(durations)
    summary = {
        "format": "cs336-tokenizer-benchmark-v1",
        "created_at_utc": utc_timestamp(),
        "tokenizer_artifact": metadata_path.name,
        "tokenizer_vocab_size": metadata["vocab_size"],
        "tokenizer_sha256": metadata.get("sha256"),
        "source_filename": args.input.name,
        "characters_per_pass": char_count,
        "bytes_per_pass": byte_count,
        "tokens_per_pass": token_count,
        "compression_bytes_per_token": byte_count / token_count if token_count else None,
        "repeat": args.repeat,
        "durations_seconds": durations,
        "median_seconds": median_seconds,
        "median_throughput_bytes_per_second": byte_count / median_seconds if median_seconds else None,
        "median_throughput_tokens_per_second": token_count / median_seconds if median_seconds else None,
        "throughput_bytes_per_second": total_bytes / total_seconds if total_seconds else None,
        "throughput_tokens_per_second": token_count * args.repeat / total_seconds if total_seconds else None,
        "peak_rss_bytes": memory.peak_bytes,
    }
    if args.output is not None:
        atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
