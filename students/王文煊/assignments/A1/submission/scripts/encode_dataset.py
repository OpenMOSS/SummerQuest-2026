"""Encode a text corpus into a flat uint16 array of token ids (memory friendly).

Documents are joined by the ``<|endoftext|>`` token id so the model sees the
document boundary. Uses ``encode_iterable`` to keep memory bounded.

Example:
    python scripts/encode_dataset.py \
        --tokenizer artifacts/tinystories_tokenizer.pkl \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --out artifacts/tinystories_train.npy
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from array import array

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return Tokenizer(data["vocab"], data["merges"], data["special_tokens"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--report-every", type=int, default=5_000_000)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    dtype = np.dtype(args.dtype)

    start = time.time()
    ids = array("H") if dtype == np.uint16 else array("l")
    count = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
            count += 1
            if count % args.report_every == 0:
                elapsed = time.time() - start
                print(f"[encode] {count:,} tokens, {count / elapsed / 1e3:.1f}k tok/s", flush=True)

    arr = np.frombuffer(ids, dtype=dtype).copy()
    np.save(args.out, arr)

    n_bytes = os.path.getsize(args.input)
    elapsed = time.time() - start
    print(f"[encode] input={args.input}")
    print(f"[encode] num_tokens={len(arr):,}")
    print(f"[encode] source_bytes={n_bytes:,}")
    print(f"[encode] compression_ratio_bytes_per_token={n_bytes / len(arr):.3f}")
    print(f"[encode] throughput_bytes_per_sec={n_bytes / elapsed:,.0f}")
    print(f"[encode] elapsed_sec={elapsed:.1f}")
    print(f"[encode] saved -> {args.out} dtype={dtype}")


if __name__ == "__main__":
    main()
