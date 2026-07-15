from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from cs336_basics.tokenizer import Tokenizer


_WORKER_TOKENIZER: Tokenizer | None = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def initialize_worker(tokenizer_path: str, cache_size: int) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.load(tokenizer_path)
    _WORKER_TOKENIZER.enable_merge_cache(cache_size)


def benchmark_chunk(task: tuple[str, int, int]) -> tuple[int, int]:
    input_path, start, end = task
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("benchmark worker tokenizer was not initialized")

    byte_count = 0
    token_count = 0
    with Path(input_path).open("rb") as source:
        if start == 0:
            source.seek(0)
        else:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                source.readline()

        while source.tell() < end:
            encoded_line = source.readline()
            if not encoded_line:
                break
            byte_count += len(encoded_line)
            token_count += len(tokenizer.encode(encoded_line.decode("utf-8")))

    return byte_count, token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure tokenizer compression and encoding throughput.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true", help="Show input-byte progress on stderr.")
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Number of tokenizer worker processes (default: 1).",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=positive_int,
        default=16,
        help="Approximate input chunk size for multiprocessing (default: 16 MiB).",
    )
    parser.add_argument(
        "--cache-size",
        type=positive_int,
        default=32768,
        help="Repeated pre-token BPE cache entries per worker (default: 32768).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.load(args.tokenizer)
    byte_count = 0
    token_count = 0
    input_size = args.input.stat().st_size
    chunk_size = args.chunk_size_mb * 1024 * 1024
    chunks = [
        (str(args.input), start, min(start + chunk_size, input_size)) for start in range(0, input_size, chunk_size)
    ]
    worker_count = min(args.workers, len(chunks)) if chunks else 1
    started = time.perf_counter()
    with tqdm(
        total=input_size,
        desc=f"Tokenizer benchmark ({worker_count} worker{'s' if worker_count != 1 else ''})",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not args.progress,
    ) as progress:
        if worker_count == 1:
            initialize_worker(str(args.tokenizer), args.cache_size)
            for chunk in chunks:
                chunk_bytes, chunk_tokens = benchmark_chunk(chunk)
                byte_count += chunk_bytes
                token_count += chunk_tokens
                progress.update(chunk[2] - chunk[1])
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=initialize_worker,
                initargs=(str(args.tokenizer), args.cache_size),
            ) as executor:
                futures = {executor.submit(benchmark_chunk, chunk): chunk for chunk in chunks}
                for future in as_completed(futures):
                    chunk = futures[future]
                    chunk_bytes, chunk_tokens = future.result()
                    byte_count += chunk_bytes
                    token_count += chunk_tokens
                    progress.update(chunk[2] - chunk[1])
    elapsed = time.perf_counter() - started
    longest_id, longest_bytes = max(tokenizer.vocab.items(), key=lambda item: len(item[1]))
    result = {
        "input_name": args.input.name,
        "workers": worker_count,
        "cache_size": args.cache_size,
        "bytes": byte_count,
        "tokens": token_count,
        "bytes_per_token": byte_count / token_count if token_count else None,
        "tokens_per_sec": token_count / elapsed if elapsed else None,
        "megabytes_per_sec": byte_count / elapsed / 1_000_000 if elapsed else None,
        "longest_token_id": longest_id,
        "longest_token_bytes": len(longest_bytes),
        "longest_token_preview": longest_bytes.decode("utf-8", errors="replace"),
        "elapsed_sec": elapsed,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
