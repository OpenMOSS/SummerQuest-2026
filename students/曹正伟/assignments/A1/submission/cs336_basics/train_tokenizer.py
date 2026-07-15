"""Command-line entry point for training and serializing a byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import psutil

from cs336_basics.bpe import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be greater than 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    stop_sampling = threading.Event()
    peak_rss = process.memory_info().rss

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop_sampling.wait(0.05):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    started = time.perf_counter()
    try:
        vocab, merges = train_bpe(
            args.input_path,
            args.vocab_size,
            args.special_token,
            workers=args.workers,
        )
    finally:
        stop_sampling.set()
        sampler.join()
        peak_rss = max(peak_rss, process.memory_info().rss)

    elapsed = time.perf_counter() - started

    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.json"
    metadata_path = args.output_dir / "metadata.json"

    vocab_path.write_text(
        json.dumps({str(token_id): token.hex() for token_id, token in vocab.items()}, indent=2),
        encoding="utf-8",
    )
    merges_path.write_text(
        json.dumps([[left.hex(), right.hex()] for left, right in merges], indent=2),
        encoding="utf-8",
    )

    special_bytes = {token.encode("utf-8") for token in args.special_token}
    ordinary_vocab = [item for item in vocab.items() if item[1] not in special_bytes]
    longest_id, longest_token = max(ordinary_vocab or list(vocab.items()), key=lambda item: len(item[1]))

    metadata = {
        "input_path": str(args.input_path),
        "workers": args.workers,
        "requested_vocab_size": args.vocab_size,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": args.special_token,
        "training_seconds": elapsed,
        "elapsed_seconds": elapsed,
        "peak_process_rss_bytes": peak_rss,
        "longest_token_id": longest_id,
        "longest_token_hex": longest_token.hex(),
        "longest_token_utf8": longest_token.decode("utf-8", errors="replace"),
        "longest_token_bytes": len(longest_token),
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()