#!/usr/bin/env python3
"""Stream a UTF-8 corpus into a memory-mappable one-dimensional ``.npy`` array."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=["auto", "uint16", "uint32", "int64"], default="auto")
    parser.add_argument("--buffer-tokens", type=int, default=1_000_000)
    return parser.parse_args()


def choose_dtype(requested: str, max_token_id: int) -> np.dtype:
    if requested == "auto":
        return np.dtype(np.uint16 if max_token_id <= np.iinfo(np.uint16).max else np.uint32)
    dtype = np.dtype(requested)
    if np.issubdtype(dtype, np.unsignedinteger) and max_token_id > np.iinfo(dtype).max:
        raise ValueError(f"token ID {max_token_id} does not fit in {dtype}")
    return dtype


def main() -> None:
    args = parse_args()
    if args.buffer_tokens <= 0:
        raise ValueError("--buffer-tokens must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    tokenizer = Tokenizer.from_directory(args.tokenizer_dir)
    max_token_id = max(tokenizer.vocab)
    dtype = choose_dtype(args.dtype, max_token_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    token_count = 0
    bytes_read = args.input.stat().st_size
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=args.output.name + ".",
            suffix=".tokens.tmp",
            dir=args.output.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            buffer: list[int] = []
            with args.input.open(encoding="utf-8", newline="") as input_file:
                for token_id in tokenizer.encode_iterable(input_file):
                    buffer.append(token_id)
                    if len(buffer) >= args.buffer_tokens:
                        array = np.asarray(buffer, dtype=dtype)
                        array.tofile(temporary_file)
                        token_count += len(buffer)
                        buffer.clear()
            if buffer:
                array = np.asarray(buffer, dtype=dtype)
                array.tofile(temporary_file)
                token_count += len(buffer)

        output = np.lib.format.open_memmap(args.output, mode="w+", dtype=dtype, shape=(token_count,))
        copied = 0
        with temporary_path.open("rb") as temporary_file:
            while copied < token_count:
                count = min(args.buffer_tokens, token_count - copied)
                chunk = np.fromfile(temporary_file, dtype=dtype, count=count)
                if len(chunk) != count:
                    raise OSError("temporary token stream ended unexpectedly")
                output[copied : copied + count] = chunk
                copied += count
        output.flush()
        del output
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    report = {
        "input": str(args.input),
        "tokenizer_dir": str(args.tokenizer_dir),
        "output": str(args.output),
        "dtype": dtype.name,
        "token_count": token_count,
        "input_bytes": bytes_read,
        "bytes_per_token": bytes_read / token_count if token_count else None,
        "elapsed_seconds": elapsed,
        "input_bytes_per_second": bytes_read / elapsed if elapsed else None,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, sort_keys=True)
        report_file.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
