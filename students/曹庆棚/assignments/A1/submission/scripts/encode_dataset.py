from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from cs336_basics.tokenizer import Tokenizer


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a UTF-8 corpus into a memory-mappable NumPy token array.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--progress", action="store_true", help="Show both encoding passes on stderr.")
    parser.add_argument(
        "--cache-size",
        type=nonnegative_int,
        default=32768,
        help="Number of repeated pre-token BPE results to cache; 0 disables caching (default: 32768).",
    )
    return parser.parse_args()


def track_input_bytes(lines: Iterable[str], progress: tqdm) -> Iterator[str]:
    for text in lines:
        yield text
        progress.update(len(text.encode("utf-8")))


def count_tokens(tokenizer: Tokenizer, input_path: Path, *, show_progress: bool) -> int:
    count = 0
    with (
        input_path.open(encoding="utf-8") as source,
        tqdm(
            total=input_path.stat().st_size,
            desc="Encode pass 1/2 (count)",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress,
    ):
        for token_id in tokenizer.encode_iterable(track_input_bytes(source, progress)):
            del token_id
            count += 1
    return count


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.load(args.tokenizer)
    tokenizer.enable_merge_cache(args.cache_size)
    dtype = np.uint16 if max(tokenizer.vocab) <= np.iinfo(np.uint16).max else np.uint32

    started = time.perf_counter()
    token_count = count_tokens(tokenizer, args.input, show_progress=args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(args.output, mode="w+", dtype=dtype, shape=(token_count,))

    offset = 0
    buffer: list[int] = []
    buffer_size = 1_000_000
    with (
        args.input.open(encoding="utf-8") as source,
        tqdm(
            total=args.input.stat().st_size,
            desc="Encode pass 2/2 (write)",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not args.progress,
        ) as progress,
    ):
        for token_id in tokenizer.encode_iterable(track_input_bytes(source, progress)):
            buffer.append(token_id)
            if len(buffer) >= buffer_size:
                output[offset : offset + len(buffer)] = np.asarray(buffer, dtype=dtype)
                offset += len(buffer)
                buffer.clear()
    if buffer:
        output[offset : offset + len(buffer)] = np.asarray(buffer, dtype=dtype)
        offset += len(buffer)
    output.flush()
    elapsed = time.perf_counter() - started

    result = {
        "input_name": args.input.name,
        "output_name": args.output.name,
        "tokens": offset,
        "dtype": np.dtype(dtype).name,
        "cache_size": args.cache_size,
        "elapsed_sec": elapsed,
        "tokens_per_sec": offset / elapsed if elapsed else None,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
