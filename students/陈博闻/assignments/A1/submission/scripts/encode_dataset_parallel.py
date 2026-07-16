from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.bpe import Tokenizer
from scripts.encode_dataset import load_hex_merges, load_hex_vocab


_TOKENIZER = None


def init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global _TOKENIZER
    _TOKENIZER = Tokenizer(load_hex_vocab(Path(vocab_path)), load_hex_merges(Path(merges_path)), special_tokens)


def encode_chunk(item: tuple[int, str]) -> tuple[int, np.ndarray]:
    index, text = item
    assert _TOKENIZER is not None
    ids = _TOKENIZER.encode(text)
    return index, np.asarray(ids, dtype=np.uint16)


def iter_docs(path: Path, special_token: str, chunk_docs: int):
    buffer = []
    doc_count = 0
    index = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            buffer.append(line)
            doc_count += line.count(special_token)
            if doc_count >= chunk_docs:
                yield index, "".join(buffer)
                index += 1
                buffer = []
                doc_count = 0
    if buffer:
        yield index, "".join(buffer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiprocess encode text into a NumPy token-id array.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-docs", type=int, default=256)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    start = time.time()
    input_path = Path(args.input)
    output_path = Path(args.output)
    tmp_dir = output_path.with_suffix(output_path.suffix + ".parts")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    special_token = args.special_token[0]
    completed = 0
    total_tokens = 0
    parts: list[Path] = []

    with mp.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.vocab, args.merges, args.special_token),
    ) as pool:
        for index, arr in pool.imap(encode_chunk, iter_docs(input_path, special_token, args.chunk_docs), chunksize=1):
            part_path = tmp_dir / f"{index:08d}.npy"
            np.save(part_path, arr)
            parts.append(part_path)
            completed += 1
            total_tokens += int(arr.size)
            if completed % 20 == 0:
                elapsed = time.time() - start
                print(json.dumps({
                    "chunks": completed,
                    "tokens": total_tokens,
                    "elapsed_sec": round(elapsed, 1),
                    "tokens_per_sec": round(total_tokens / elapsed, 1) if elapsed > 0 else None,
                }), flush=True)

    parts.sort()
    arrays = [np.load(path, mmap_mode="r") for path in parts]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(sum(arr.size for arr in arrays),))
    offset = 0
    for arr in arrays:
        out[offset : offset + arr.size] = arr
        offset += arr.size
    out.flush()

    elapsed = time.time() - start
    print(json.dumps({
        "output": str(output_path),
        "chunks": completed,
        "tokens": int(offset),
        "elapsed_sec": round(elapsed, 1),
        "tokens_per_sec": round(offset / elapsed, 1) if elapsed > 0 else None,
    }), flush=True)


if __name__ == "__main__":
    main()
