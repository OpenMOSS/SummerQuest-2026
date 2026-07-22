from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer_io import load_tokenizer


_WORKER_TOKENIZER = None


def initialize_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_tokenizer(tokenizer_path)


def encode_text(text: str) -> list[int]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return _WORKER_TOKENIZER.encode(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream text into a raw token-ID binary file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    tokenizer = load_tokenizer(args.tokenizer)
    dtype = np.dtype(args.dtype)
    if max(tokenizer.vocab) > np.iinfo(dtype).max:
        raise ValueError(f"tokenizer IDs do not fit in {dtype}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    start = time.perf_counter()
    with open(args.input, encoding="utf-8") as source, open(output, "wb") as target:
        buffer: list[int] = []
        if args.workers == 1:
            encoded_lines = (tokenizer.encode(line) for line in source)
            pool = None
        else:
            pool = mp.Pool(
                args.workers,
                initializer=initialize_worker,
                initargs=(args.tokenizer,),
            )
            encoded_lines = pool.imap(encode_text, source, chunksize=256)
        for token_ids in encoded_lines:
            buffer.extend(token_ids)
            if len(buffer) >= 1_000_000:
                np.asarray(buffer, dtype=dtype).tofile(target)
                count += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(target)
            count += len(buffer)
        if pool is not None:
            pool.close()
            pool.join()
    elapsed = time.perf_counter() - start
    metadata = {"tokens": count, "dtype": args.dtype, "seconds": elapsed}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"encoded tokens={count} tokens_per_sec={count / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    main()
