from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import resource
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_DTYPE: np.dtype | None = None


def _init_worker(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    dtype_name: str,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_DTYPE
    _WORKER_TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    _WORKER_DTYPE = np.dtype(dtype_name)


def _encode_chunk(args: tuple[str, int, int]) -> tuple[bytes, int]:
    if _WORKER_TOKENIZER is None or _WORKER_DTYPE is None:
        raise RuntimeError("encoding worker was not initialized")
    input_path, start, end = args
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors="ignore")
    token_ids = _WORKER_TOKENIZER.encode(text)
    return np.asarray(token_ids, dtype=_WORKER_DTYPE).tobytes(), len(token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-tokenize a text corpus into a flat binary token array.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--merges", required=True, type=Path)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--output", required=True, type=Path, help="Output .bin path")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--num-processes", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    dtype = np.uint16 if max(tokenizer.vocab) <= np.iinfo(np.uint16).max else np.uint32
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    token_count = 0
    digest = hashlib.sha256()
    input_size = args.input.stat().st_size
    use_parallel = input_size > 64 * 1024 * 1024 and args.num_processes > 1 and args.special_token
    with args.output.open("wb") as destination:
        if use_parallel:
            with args.input.open("rb") as source:
                boundaries = find_chunk_boundaries(
                    source,
                    args.num_processes * 8,
                    args.special_token[0].encode("utf-8"),
                )
            tasks = [
                (str(args.input), start_offset, end_offset)
                for start_offset, end_offset in zip(boundaries[:-1], boundaries[1:])
                if end_offset > start_offset
            ]
            with mp.get_context("fork").Pool(
                processes=min(args.num_processes, len(tasks)),
                initializer=_init_worker,
                initargs=(str(args.vocab), str(args.merges), args.special_token, np.dtype(dtype).name),
            ) as pool:
                for encoded_bytes, chunk_token_count in pool.imap(_encode_chunk, tasks):
                    destination.write(encoded_bytes)
                    digest.update(encoded_bytes)
                    token_count += chunk_token_count
        else:
            buffer: list[int] = []
            with args.input.open(encoding="utf-8") as source:
                for token_id in tokenizer.encode_iterable(source):
                    buffer.append(token_id)
                    if len(buffer) >= args.chunk_size:
                        encoded_bytes = np.asarray(buffer, dtype=dtype).tobytes()
                        destination.write(encoded_bytes)
                        digest.update(encoded_bytes)
                        token_count += len(buffer)
                        buffer.clear()
                if buffer:
                    encoded_bytes = np.asarray(buffer, dtype=dtype).tobytes()
                    destination.write(encoded_bytes)
                    digest.update(encoded_bytes)
                    token_count += len(buffer)
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "dtype": np.dtype(dtype).name,
        "token_count": token_count,
        "source_bytes": input_size,
        "compression_ratio_bytes_per_token": input_size / max(token_count, 1),
        "wall_clock_sec": time.perf_counter() - start,
        "throughput_bytes_per_sec": input_size / max(time.perf_counter() - start, 1e-9),
        "num_processes": args.num_processes if use_parallel else 1,
        "output_sha256": digest.hexdigest(),
        "peak_rss_mb": max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        / 1024,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
