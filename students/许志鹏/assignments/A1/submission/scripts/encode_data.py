from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.tokenizer import Tokenizer

_TOKENIZER: Tokenizer | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode UTF-8 text into a 1D NumPy token ID array.")
    parser.add_argument("--input", required=True, help="Path to UTF-8 text.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer pickle from scripts/train_tokenizer.py.")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument("--dtype", choices=["uint16", "uint32", "int64"], default="uint16")
    parser.add_argument("--metadata", default=None, help="Optional JSON metadata output path.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for encoding.")
    parser.add_argument(
        "--split-mode",
        choices=["special", "line", "none"],
        default="special",
        help="How to split text before parallel encoding. Use 'special' for TinyStories/OWT.",
    )
    return parser.parse_args()


def load_tokenizer(path: str | Path) -> Tokenizer:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return Tokenizer(payload["vocab"], payload["merges"], payload.get("special_tokens"))


def load_tokenizer_payload(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def init_worker(payload: dict) -> None:
    global _TOKENIZER
    _TOKENIZER = Tokenizer(payload["vocab"], payload["merges"], payload.get("special_tokens"))


def encode_part(text: str) -> list[int]:
    if _TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    return _TOKENIZER.encode(text)


def chunk_items(items: list[str], num_chunks: int) -> list[str]:
    if not items:
        return []
    chunk_size = max(1, math.ceil(len(items) / num_chunks))
    return ["".join(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def split_text(text: str, tokenizer: Tokenizer, split_mode: str, num_workers: int) -> list[str]:
    if num_workers <= 1 or split_mode == "none":
        return [text]
    if split_mode == "line":
        return chunk_items(text.splitlines(keepends=True), num_workers * 8)
    if tokenizer.special_pattern is None:
        return [text]
    parts = [part for part in tokenizer.special_pattern.split(text) if part]
    return chunk_items(parts, num_workers * 8)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = load_tokenizer_payload(args.tokenizer)
    tokenizer = Tokenizer(payload["vocab"], payload["merges"], payload.get("special_tokens"))

    start = time.time()
    text = Path(args.input).read_text(encoding="utf-8")
    parts = split_text(text, tokenizer, args.split_mode, args.num_workers)
    if args.num_workers > 1 and len(parts) > 1:
        with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(payload,)) as pool:
            encoded_parts = pool.map(encode_part, parts)
        token_ids = [token_id for part in encoded_parts for token_id in part]
    else:
        token_ids = tokenizer.encode(text)
    elapsed = time.time() - start

    max_id = max(token_ids) if token_ids else 0
    dtype = np.dtype(args.dtype)
    if max_id > np.iinfo(dtype).max:
        raise ValueError(f"max token id {max_id} does not fit in dtype {dtype}")
    array = np.asarray(token_ids, dtype=dtype)
    np.save(output, array)

    input_bytes = Path(args.input).stat().st_size
    metadata = {
        "input": str(args.input),
        "tokenizer": str(args.tokenizer),
        "output": str(output),
        "dtype": str(dtype),
        "num_tokens": int(array.size),
        "max_token_id": int(max_id),
        "input_bytes": int(input_bytes),
        "bytes_per_token": float(input_bytes / array.size) if array.size else None,
        "tokens_per_sec": float(array.size / elapsed) if elapsed > 0 else None,
        "num_workers": args.num_workers,
        "split_mode": args.split_mode,
        "elapsed_sec": elapsed,
    }
    if args.metadata is not None:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
