"""Tokenizer experiments for problem tokenizer_experiments.

(a) Compression ratio (bytes/token) for TS/OWT docs with TS/OWT tokenizers
(b) Qualitative comparison (printed for analysis; write the answer in writeup)
(c) Throughput (bytes/s) and estimated time to tokenize Pile (825 GB)

Requires both tokenizers to be trained first:
    uv run python scripts/train_bpe_tinystories.py
    uv run python scripts/train_bpe_owt.py

Usage:
    uv run python scripts/tokenizer_experiments.py
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

from cs336_basics.bpe import BPETokenizer
from cs336_basics.tokenizer_parallel import (
    enable_encode_cache,
    find_special_token_aligned_ranges,
    iter_tokenizable_segments_in_range,
)

TS_VOCAB = Path("data/tinystories_vocab.json")
TS_MERGES = Path("data/tinystories_merges.txt")
OWT_VOCAB = Path("data/owt_vocab.json")
OWT_MERGES = Path("data/owt_merges.txt")

TS_TRAIN = Path("data/TinyStoriesV2-GPT4-train.txt")
OWT_TRAIN = Path("data/owt_train.txt")

PILE_SIZE_BYTES = 825 * 10**9
END_OF_TEXT = "<|endoftext|>"
DOCUMENT_READ_CHUNK_SIZE = 1024 * 1024
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_CACHE_SIZE = 65_536

_WORKER_TOKENIZER: BPETokenizer | None = None


def load_tokenizer(vocab_path: Path, merges_path: Path, cache_size: int = 0) -> BPETokenizer:
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    enable_encode_cache(tokenizer, cache_size)
    return tokenizer


def sample_docs(path: Path, n: int = 10) -> list[str]:
    if n <= 0:
        return []

    docs: list[str] = []
    pending = ""
    with open(path, encoding="utf-8") as file:
        while len(docs) < n:
            chunk = file.read(DOCUMENT_READ_CHUNK_SIZE)
            if not chunk:
                break

            pieces = (pending + chunk).split(END_OF_TEXT)
            pending = pieces.pop()
            for document in pieces:
                if document.strip():
                    docs.append(document)
                    if len(docs) == n:
                        return docs

    if pending.strip() and len(docs) < n:
        docs.append(pending)
    return docs


def compression_ratio(tok: BPETokenizer, docs: list[str]) -> float:
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tok.encode(d)) for d in docs)
    if total_tokens == 0:
        raise ValueError("cannot compute compression ratio from an empty sample")
    return total_bytes / total_tokens


def part_a(ts_tok: BPETokenizer, owt_tok: BPETokenizer, ts_docs: list[str], owt_docs: list[str]) -> None:
    print("=== (a) Compression ratio (bytes/token) ===")
    print(f"{'Dataset':<20} {'Tokenizer':<15} {'bytes/token':>12}")
    results: dict[tuple[str, str], float] = {}
    for dataset_label, docs in [("TS docs", ts_docs), ("OWT docs", owt_docs)]:
        for tok_label, tok in [("TS 10K", ts_tok), ("OWT 32K", owt_tok)]:
            ratio = compression_ratio(tok, docs)
            results[(dataset_label, tok_label)] = ratio
            print(f"{dataset_label:<20} {tok_label:<15} {ratio:>12.2f}")

    print("\n=== (b) Qualitative (analyze from above) ===")
    ts_on_owt = results[("OWT docs", "TS 10K")]
    ts_on_ts = results[("TS docs", "TS 10K")]
    print(f"TS tokenizer on TS docs: {ts_on_ts:.2f}")
    print(f"TS tokenizer on OWT docs: {ts_on_owt:.2f}")
    print(f"Ratio drop: {(ts_on_ts - ts_on_owt) / ts_on_ts * 100:.1f}%")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _init_benchmark_worker(vocab_path: str, merges_path: str, cache_size: int) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_tokenizer(Path(vocab_path), Path(merges_path), cache_size)


def _benchmark_range(task: tuple[int, Path, int, int]) -> tuple[int, int, int]:
    index, input_path, start, end = task
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("benchmark worker was not initialized")

    segments = iter_tokenizable_segments_in_range(input_path, start, end, END_OF_TEXT)
    total_tokens = sum(len(_WORKER_TOKENIZER.encode(segment)) for segment in segments)
    return index, end - start, total_tokens


def part_c(workers: int, chunks_per_worker: int, throughput_mib: float | None, cache_size: int) -> None:
    print("\n=== (c) Throughput and Pile estimate ===")
    if not OWT_TRAIN.is_file():
        raise FileNotFoundError(OWT_TRAIN)

    max_bytes = None if throughput_mib is None else int(throughput_mib * 1024**2)
    ranges = find_special_token_aligned_ranges(
        OWT_TRAIN,
        workers * chunks_per_worker,
        END_OF_TEXT,
        max_bytes=max_bytes,
    )
    if not ranges:
        raise ValueError(f"cannot benchmark empty dataset: {OWT_TRAIN}")

    tasks = [(index, OWT_TRAIN, start, end) for index, (start, end) in enumerate(ranges)]
    total_bytes = 0
    total_tokens = 0
    t0 = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        min(workers, len(tasks)),
        initializer=_init_benchmark_worker,
        initargs=(str(OWT_VOCAB), str(OWT_MERGES), cache_size),
    ) as pool:
        for completed, (index, byte_count, token_count) in enumerate(
            pool.imap_unordered(_benchmark_range, tasks), start=1
        ):
            total_bytes += byte_count
            total_tokens += token_count
            print(f"  ranges: {completed}/{len(tasks)} (finished {index})", end="\r", flush=True)
    elapsed = time.perf_counter() - t0
    print(" " * 72, end="\r")

    throughput = total_bytes / elapsed
    pile_hours = PILE_SIZE_BYTES / throughput / 3600

    print(f"Workers: {workers}; processed: {total_bytes / 1024**2:.0f} MiB in {elapsed:.1f}s")
    print(f"Tokens: {total_tokens:,}; throughput: {throughput / 1024**2:.1f} MiB/s")
    print(f"Pile (825 GB decimal) estimated: {pile_hours:.1f} hours")


def main(workers: int, chunks_per_worker: int, throughput_mib: float | None, cache_size: int) -> None:
    ts_tok = load_tokenizer(TS_VOCAB, TS_MERGES, cache_size)
    owt_tok = load_tokenizer(OWT_VOCAB, OWT_MERGES, cache_size)

    ts_docs = sample_docs(TS_TRAIN, n=10)
    owt_docs = sample_docs(OWT_TRAIN, n=10)

    part_a(ts_tok, owt_tok, ts_docs, owt_docs)
    part_c(workers, chunks_per_worker, throughput_mib, cache_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunks-per-worker", type=_positive_int, default=4)
    parser.add_argument(
        "--throughput-mib",
        type=_positive_float,
        default=None,
        help="Benchmark only an approximately sized prefix; default is the full OWT training file.",
    )
    parser.add_argument(
        "--cache-size",
        type=_nonnegative_int,
        default=DEFAULT_CACHE_SIZE,
        help="Per-worker BPE pre-token cache entries; use 0 to disable (default: 65536).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.workers, args.chunks_per_worker, args.throughput_mib, args.cache_size)
