#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import Tokenizer, _find_chunk_boundaries


def encode_chunk(job: tuple) -> tuple[int, int]:
    index, input_path, start, end, vocab_path, merges_path, special_tokens, dtype_name, shard_path = job
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, list(special_tokens))
    dtype = np.dtype(dtype_name)
    count = 0
    with open(input_path, "rb") as source, open(shard_path, "wb") as target:
        source.seek(start)

        def text_chunks():
            while source.tell() < end:
                remaining = end - source.tell()
                line = source.readline(remaining)
                if not line:
                    break
                yield line.decode("utf-8")

        token_buffer: list[int] = []
        for token_id in tokenizer.encode_iterable(text_chunks()):
            token_buffer.append(token_id)
            if len(token_buffer) >= 1_000_000:
                np.asarray(token_buffer, dtype=dtype).tofile(target)
                count += len(token_buffer)
                token_buffer.clear()
        if token_buffer:
            np.asarray(token_buffer, dtype=dtype).tofile(target)
            count += len(token_buffer)
    return index, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode UTF-8 text as a memory-mappable token array.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    dtype = np.uint16 if len(tokenizer.vocab) <= np.iinfo(np.uint16).max else np.uint32
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / f".{output.name}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    split_token = args.special_token[0].encode("utf-8") if args.special_token else b""
    with open(args.input, "rb") as source:
        boundaries = _find_chunk_boundaries(source, args.num_processes, split_token)
    jobs = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        jobs.append(
            (
                index,
                args.input,
                start,
                end,
                args.vocab,
                args.merges,
                tuple(args.special_token),
                np.dtype(dtype).name,
                str(shard_dir / f"{index:04d}.bin"),
            )
        )
    start = time.perf_counter()
    counts: dict[int, int] = {}
    with ProcessPoolExecutor(max_workers=min(args.num_processes, len(jobs))) as executor:
        for index, count in executor.map(encode_chunk, jobs):
            counts[index] = count
    with open(output, "wb") as target:
        for index in range(len(jobs)):
            shard_path = shard_dir / f"{index:04d}.bin"
            with open(shard_path, "rb") as shard:
                shutil.copyfileobj(shard, target, length=16 << 20)
            shard_path.unlink()
    shard_dir.rmdir()
    count = sum(counts.values())
    elapsed = time.perf_counter() - start
    metadata = {
        "source": Path(args.input).name,
        "tokens": count,
        "dtype": np.dtype(dtype).name,
        "elapsed_sec": elapsed,
        "tokens_per_sec": count / elapsed,
        "num_processes": min(args.num_processes, len(jobs)),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
