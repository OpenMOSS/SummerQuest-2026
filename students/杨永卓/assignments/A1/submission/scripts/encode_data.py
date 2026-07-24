#!/usr/bin/env python3
"""Stream text through a tokenizer and write uint16 token IDs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from cs336_basics.experiment import save_json
from cs336_basics.tokenizer import Tokenizer


_WORKER_TOKENIZER: Tokenizer | None = None


def initialize_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.load(tokenizer_path)


def encode_lines(lines: list[str]) -> list[int]:
    assert _WORKER_TOKENIZER is not None
    token_ids: list[int] = []
    for line in lines:
        token_ids.extend(_WORKER_TOKENIZER.encode(line))
    return token_ids


def line_chunks(source, target_characters: int):
    chunk: list[str] = []
    character_count = 0
    for line in source:
        chunk.append(line)
        character_count += len(line)
        if character_count >= target_characters:
            yield chunk
            chunk = []
            character_count = 0
    if chunk:
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-characters", type=int, default=2_000_000)
    args = parser.parse_args()
    tokenizer = Tokenizer.load(args.tokenizer)
    if len(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary does not fit in uint16")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    start = time.perf_counter()
    with open(args.input, encoding="utf-8") as source, open(destination, "wb") as target:
        chunks = line_chunks(source, args.chunk_characters)
        if args.workers == 1:
            initialize_worker(args.tokenizer)
            encoded_chunks = map(encode_lines, chunks)
            pool = None
        else:
            pool = mp.Pool(args.workers, initializer=initialize_worker, initargs=(args.tokenizer,))
            encoded_chunks = pool.imap(encode_lines, chunks, chunksize=1)
        try:
            for chunk_index, token_ids in enumerate(encoded_chunks, start=1):
                np.asarray(token_ids, dtype=np.uint16).tofile(target)
                token_count += len(token_ids)
                if chunk_index % 10 == 0:
                    print(f"encoded chunks={chunk_index}, tokens={token_count}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    elapsed = time.perf_counter() - start
    byte_count = Path(args.input).stat().st_size
    save_json(
        args.summary,
        {
            "input_bytes": byte_count,
            "token_count": token_count,
            "compression_ratio_bytes_per_token": byte_count / token_count,
            "elapsed_sec": elapsed,
            "throughput_bytes_per_sec": byte_count / elapsed,
        },
    )
    print(f"encoded {token_count} tokens in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
