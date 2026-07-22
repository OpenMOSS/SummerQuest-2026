from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import load_tokenizer, load_tokenizer_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text file into a NumPy token ID array.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--dtype", choices=["auto", "uint16", "uint32", "int64"], default="auto")
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--progress-every-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()

    vocab, _, _ = load_tokenizer_spec(args.tokenizer)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "uint16" if len(vocab) <= np.iinfo(np.uint16).max + 1 else "uint32"

    tokenizer = load_tokenizer(args.tokenizer)
    start = time.perf_counter()
    byte_count = 0

    def iter_limited_lines():
        nonlocal byte_count
        next_report = args.progress_every_bytes
        with open(args.input, encoding="utf-8") as f:
            for chunk in f:
                if args.max_bytes is not None and byte_count >= args.max_bytes:
                    break
                if args.max_bytes is not None:
                    encoded_chunk = chunk.encode("utf-8")
                    remaining = args.max_bytes - byte_count
                    if len(encoded_chunk) > remaining:
                        chunk = encoded_chunk[:remaining].decode("utf-8", errors="ignore")
                byte_count += len(chunk.encode("utf-8"))
                if byte_count >= next_report:
                    print(
                        json.dumps(
                            {
                                "stage": "encode",
                                "bytes": byte_count,
                                "wall_clock_sec": time.perf_counter() - start,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    while next_report <= byte_count:
                        next_report += args.progress_every_bytes
                yield chunk

    token_ids = np.fromiter(tokenizer.encode_iterable(iter_limited_lines()), dtype=np.dtype(dtype))
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, token_ids)

    if args.max_bytes is None:
        byte_count = args.input.stat().st_size
    summary = {
        "tokenizer_path": str(args.tokenizer),
        "input_path": str(args.input),
        "output_path": str(args.output),
        "dtype": dtype,
        "tokens": int(token_ids.size),
        "bytes": byte_count,
        "max_bytes": args.max_bytes or 0,
        "compression_ratio_bytes_per_token": byte_count / int(token_ids.size) if token_ids.size else 0,
        "encode_wall_clock_sec": elapsed,
        "throughput_tokens_per_sec": int(token_ids.size) / elapsed if elapsed else 0,
        "throughput_bytes_per_sec": byte_count / elapsed if elapsed else 0,
    }

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
