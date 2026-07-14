from __future__ import annotations

import argparse
import resource
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.artifacts import load_tokenizer
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode UTF-8 text into a memory-mapped NumPy token array.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def iter_ids(tokenizer: Tokenizer, path: Path) -> Iterator[int]:
    with path.open(encoding="utf-8") as source:
        with tqdm(total=path.stat().st_size, desc="tokenizing", unit="B", unit_scale=True) as progress:
            def tracked_source() -> Iterator[str]:
                for text in source:
                    progress.update(len(text.encode("utf-8")))
                    yield text

            yield from tokenizer.encode_iterable(tracked_source())


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    dtype = np.uint16 if len(tokenizer.vocab) <= np.iinfo(np.uint16).max else np.uint32

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tokens.tmp")
    token_count = 0
    buffer: list[int] = []
    buffer_size = 100_000
    print(f"encoding tokens: input={args.input} output={args.output}", flush=True)
    encode_started = time.perf_counter()
    with temporary_path.open("wb") as temporary_file:
        for token_id in iter_ids(tokenizer, args.input):
            buffer.append(token_id)
            if len(buffer) >= buffer_size:
                np.asarray(buffer, dtype=dtype).tofile(temporary_file)
                token_count += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(temporary_file)
            token_count += len(buffer)
    encode_seconds = time.perf_counter() - encode_started

    print(f"finalizing numpy array: tokens={token_count}", flush=True)
    finalize_started = time.perf_counter()
    encoded = np.lib.format.open_memmap(args.output, mode="w+", dtype=dtype, shape=(token_count,))
    raw_tokens = np.memmap(temporary_path, mode="r", dtype=dtype, shape=(token_count,))
    encoded[:] = raw_tokens
    encoded.flush()
    del raw_tokens
    temporary_path.unlink()
    finalize_seconds = time.perf_counter() - finalize_started

    bytes_read = args.input.stat().st_size
    compression_ratio = bytes_read / token_count if token_count else float("inf")
    tokens_per_second = token_count / encode_seconds if encode_seconds else float("inf")
    mib_per_second = bytes_read / (1024**2) / encode_seconds if encode_seconds else float("inf")
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"saved token ids: {args.output}")
    print(
        f"tokens={token_count} dtype={np.dtype(dtype).name} "
        f"bytes_per_token={compression_ratio:.4f} tokens_per_second={tokens_per_second:.1f} "
        f"mib_per_second={mib_per_second:.2f} encode_seconds={encode_seconds:.3f} "
        f"finalize_seconds={finalize_seconds:.3f} peak_rss_mib={peak_rss_mib:.1f}"
    )


if __name__ == "__main__":
    main()
