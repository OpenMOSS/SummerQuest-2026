"""Parallel corpus -> uint16 token-id encoder.

Splits the input file at ``<|endoftext|>`` boundaries (so documents are never
cut), encodes each chunk in a worker process, and concatenates the results in
order. Much faster than the single-process ``encode_dataset.py`` for large files.

Example:
    python scripts/encode_dataset_parallel.py \
        --tokenizer artifacts/tinystories_tokenizer.pkl \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --out artifacts/tinystories_train.npy --num-processes 64
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from array import array
from multiprocessing import Pool

import numpy as np

from cs336_basics.tokenizer import Tokenizer, find_chunk_boundaries

_TOKENIZER: Tokenizer | None = None
_PATH: str = ""


def _init(tokenizer_path: str, input_path: str) -> None:
    global _TOKENIZER, _PATH
    with open(tokenizer_path, "rb") as f:
        data = pickle.load(f)
    _TOKENIZER = Tokenizer(data["vocab"], data["merges"], data["special_tokens"])
    _PATH = input_path


def _encode_range(job) -> np.ndarray:
    idx, start, end = job
    with open(_PATH, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    buf = array("H")
    for token_id in _TOKENIZER.encode_iterable(iter(text.splitlines(keepends=True))):
        buf.append(token_id)
    return idx, np.frombuffer(buf, dtype=np.uint16).copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-processes", type=int, default=64)
    args = p.parse_args()

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_processes, b"<|endoftext|>")
    jobs = [(i, s, e) for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))]

    start = time.time()
    results: dict[int, np.ndarray] = {}
    with Pool(args.num_processes, initializer=_init, initargs=(args.tokenizer, args.input)) as pool:
        done = 0
        for idx, arr in pool.imap_unordered(_encode_range, jobs):
            results[idx] = arr
            done += 1
            print(f"[encode||] chunk {done}/{len(jobs)} ({time.time()-start:.0f}s)", flush=True)

    full = np.concatenate([results[i] for i in range(len(jobs))])
    np.save(args.out, full)

    n_bytes = os.path.getsize(args.input)
    elapsed = time.time() - start
    print(f"[encode||] input={args.input}")
    print(f"[encode||] num_tokens={len(full):,}")
    print(f"[encode||] source_bytes={n_bytes:,}")
    print(f"[encode||] compression_ratio_bytes_per_token={n_bytes / len(full):.3f}")
    print(f"[encode||] throughput_bytes_per_sec={n_bytes / elapsed:,.0f}")
    print(f"[encode||] elapsed_sec={elapsed:.1f}")
    print(f"[encode||] saved -> {args.out}")


if __name__ == "__main__":
    main()
