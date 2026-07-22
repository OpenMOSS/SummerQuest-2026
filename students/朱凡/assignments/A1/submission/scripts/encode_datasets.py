"""Encode datasets as uint16 integer sequences for problem tokenizer_experiments (d).

Encodes train and valid splits using the matching tokenizer.
Output as flat uint16 binary files for memory-mapped loading in the training stage.

Requires the matching tokenizer to be trained first:
    uv run python scripts/train_bpe_tinystories.py   # for ts
    uv run python scripts/train_bpe_owt.py            # for owt

Usage:
    uv run python scripts/encode_datasets.py ts    # encode TinyStories with TS tokenizer
    uv run python scripts/encode_datasets.py owt   # encode OpenWebText with OWT tokenizer
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.bpe import BPETokenizer
from cs336_basics.tokenizer_parallel import (
    enable_encode_cache,
    find_special_token_aligned_ranges,
    iter_tokenizable_segments_in_range,
)

TOKEN_WRITE_CHUNK_SIZE = 1_000_000
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_CACHE_SIZE = 65_536
SPECIAL_TOKENS = ["<|endoftext|>"]

_WORKER_TOKENIZER: BPETokenizer | None = None

DATASETS: dict[str, dict] = {
    "ts": {
        "tokenizer": (Path("data/tinystories_vocab.json"), Path("data/tinystories_merges.txt")),
        "splits": {
            "train": Path("data/TinyStoriesV2-GPT4-train.txt"),
            "valid": Path("data/TinyStoriesV2-GPT4-valid.txt"),
        },
        "out_prefix": "data/ts",
    },
    "owt": {
        "tokenizer": (Path("data/owt_vocab.json"), Path("data/owt_merges.txt")),
        "splits": {
            "train": Path("data/owt_train.txt"),
            "valid": Path("data/owt_valid.txt"),
        },
        "out_prefix": "data/owt",
    },
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _init_encode_worker(vocab_path: str, merges_path: str, cache_size: int) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer.from_files(vocab_path, merges_path, SPECIAL_TOKENS)
    enable_encode_cache(_WORKER_TOKENIZER, cache_size)


def _encode_range(task: tuple[int, Path, int, int, Path]) -> tuple[int, int]:
    index, input_path, start, end, part_path = task
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("encoder worker was not initialized")

    total_tokens = 0
    try:
        segments = iter_tokenizable_segments_in_range(input_path, start, end, SPECIAL_TOKENS[0])
        token_ids = _WORKER_TOKENIZER.encode_iterable(segments)
        with part_path.open("wb") as output_file:
            while True:
                chunk = np.fromiter(islice(token_ids, TOKEN_WRITE_CHUNK_SIZE), dtype=np.uint16)
                if chunk.size == 0:
                    break
                chunk.tofile(output_file)
                total_tokens += int(chunk.size)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

    return index, total_tokens


def _validate_paths(input_path: Path, vocab_path: Path, merges_path: Path) -> None:
    for path in (input_path, vocab_path, merges_path):
        if not path.is_file():
            raise FileNotFoundError(path)


def encode_split(
    vocab_path: Path,
    merges_path: Path,
    input_path: Path,
    output_path: Path,
    workers: int,
    chunks_per_worker: int,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> None:
    """Encode one split in parallel and atomically replace its output file."""
    _validate_paths(input_path, vocab_path, merges_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, SPECIAL_TOKENS)
    max_token_id = max(tokenizer.vocab, default=-1)
    if max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(f"token ID {max_token_id} does not fit in uint16")

    ranges = find_special_token_aligned_ranges(input_path, workers * chunks_per_worker, SPECIAL_TOKENS[0])
    print(f"Encoding {input_path} -> {output_path} ({workers} workers, {len(ranges)} ranges)")

    if not ranges:
        empty_file = tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
        )
        empty_path = Path(empty_file.name)
        empty_file.close()
        try:
            empty_path.replace(output_path)
        finally:
            empty_path.unlink(missing_ok=True)
        print("  0 tokens, 0.0 MiB")
        return

    total_tokens = 0
    assembled_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.parts-", dir=output_path.parent) as part_dir:
        part_paths = [Path(part_dir) / f"part-{index:06d}.bin" for index in range(len(ranges))]
        tasks = [
            (index, input_path, start, end, part_paths[index])
            for index, (start, end) in enumerate(ranges)
        ]

        context = mp.get_context("spawn")
        process_count = min(workers, len(tasks))
        with context.Pool(
            process_count,
            initializer=_init_encode_worker,
            initargs=(str(vocab_path), str(merges_path), cache_size),
        ) as pool:
            for completed, (index, token_count) in enumerate(pool.imap_unordered(_encode_range, tasks), start=1):
                total_tokens += token_count
                print(f"  ranges: {completed}/{len(tasks)} (finished {index})", end="\r", flush=True)
        print(" " * 72, end="\r")

        temporary_file = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        )
        assembled_path = Path(temporary_file.name)
        try:
            with temporary_file:
                for part_path in part_paths:
                    with part_path.open("rb") as part_file:
                        shutil.copyfileobj(part_file, temporary_file, length=16 * 1024 * 1024)
                    part_path.unlink()
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            assembled_path.replace(output_path)
            assembled_path = None
        finally:
            if assembled_path is not None:
                assembled_path.unlink(missing_ok=True)

    total_bytes = total_tokens * np.dtype(np.uint16).itemsize
    print(f"  {total_tokens} tokens, {total_bytes / 1024 / 1024:.1f} MiB")


def main(dataset: str, workers: int, chunks_per_worker: int, cache_size: int) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASETS)}")

    cfg = DATASETS[dataset]
    vocab_path, merges_path = cfg["tokenizer"]

    for split, path in cfg["splits"].items():
        out_path = Path(f"{cfg['out_prefix']}_{split}.bin")
        encode_split(vocab_path, merges_path, path, out_path, workers, chunks_per_worker, cache_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--chunks-per-worker",
        type=_positive_int,
        default=4,
        help="More chunks improve load balancing at the cost of more temporary files (default: 4).",
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
    main(args.dataset, args.workers, args.chunks_per_worker, args.cache_size)
