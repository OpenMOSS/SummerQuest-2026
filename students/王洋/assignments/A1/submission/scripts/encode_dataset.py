#!/usr/bin/env python3
"""Stream a UTF-8 corpus through a tokenizer and write a one-dimensional .npy file."""

from __future__ import annotations

import argparse
import codecs
import json
import multiprocessing
import os
import tempfile
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO

import numpy as np

from _common import atomic_write_json, iter_text_chunks, load_tokenizer_artifact, sha256_file, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a corpus without retaining the source text or all token IDs in memory.",
    )
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 input corpus")
    parser.add_argument("--tokenizer", type=Path, required=True, help="tokenizer directory or tokenizer.json")
    parser.add_argument("--output", type=Path, required=True, help="destination .npy file")
    parser.add_argument(
        "--dtype",
        choices=("auto", "uint16", "uint32"),
        default="auto",
        help="auto selects uint16 when all vocabulary IDs fit, otherwise uint32",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1024 * 1024,
        help="source characters per single-process chunk (bytes per parallel worker read)",
    )
    parser.add_argument("--buffer-tokens", type=int, default=1024 * 1024, help="token IDs buffered before a disk write")
    parser.add_argument(
        "--copy-chunk-tokens",
        type=int,
        default=8 * 1024 * 1024,
        help="IDs copied per finalization chunk",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel encoding processes; safe split points may reduce the effective count",
    )
    parser.add_argument(
        "--split-special-token",
        default="<|endoftext|>",
        help="registered special token whose UTF-8 bytes provide safe parallel split points",
    )
    parser.add_argument("--errors", choices=("strict", "replace", "ignore"), default="strict")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output and metadata sidecar")
    return parser.parse_args()


def select_dtype(requested: str, maximum_token_id: int) -> np.dtype:
    if requested == "auto":
        requested = "uint16" if maximum_token_id <= np.iinfo(np.uint16).max else "uint32"
    dtype = np.dtype(requested)
    if maximum_token_id > np.iinfo(dtype).max:
        raise ValueError(f"token ID {maximum_token_id} does not fit in {dtype.name}")
    return dtype


def metadata_path_for(output: Path) -> Path:
    return output.with_name(f"{output.name}.json")


def _find_next_boundary(
    input_file: BinaryIO,
    start: int,
    file_size: int,
    split_special_token: bytes,
    read_size: int = 1024 * 1024,
) -> int | None:
    """Find the first complete delimiter at or after ``start``."""

    input_file.seek(start)
    position = start
    overlap = b""
    overlap_size = len(split_special_token) - 1
    while position < file_size:
        block = input_file.read(min(read_size, file_size - position))
        if not block:
            break
        searchable = overlap + block
        found_at = searchable.find(split_special_token)
        if found_at != -1:
            return position - len(overlap) + found_at
        overlap = searchable[-overlap_size:] if overlap_size else b""
        position += len(block)
    return None


def find_chunk_boundaries(input_path: Path, desired_workers: int, split_special_token: bytes) -> list[int]:
    """Return ordered byte boundaries at registered special-token occurrences.

    Several target positions can resolve to the same following delimiter.  In
    that case duplicate boundaries are removed, deliberately yielding fewer
    chunks than requested instead of introducing an unsafe split.
    """

    if desired_workers <= 0:
        raise ValueError("desired_workers must be positive")
    if not split_special_token:
        raise ValueError("--split-special-token cannot be empty")

    file_size = input_path.stat().st_size
    if desired_workers == 1 or file_size == 0:
        return [0, file_size]

    targets = sorted({file_size * index // desired_workers for index in range(1, desired_workers)})
    boundaries = [0]
    last_found: int | None = None
    with input_path.open("rb") as input_file:
        for target in targets:
            if target <= 0 or target >= file_size:
                continue
            if last_found is None or last_found < target:
                last_found = _find_next_boundary(input_file, target, file_size, split_special_token)
            if last_found is None:
                # No delimiter follows this target, so none can follow a later target either.
                break
            if last_found != boundaries[-1]:
                boundaries.append(last_found)
    boundaries.append(file_size)
    return boundaries


def _validate_parallel_split(tokenizer_metadata: dict[str, object], split_special_token: str) -> None:
    """Ensure the requested delimiter is an unambiguous tokenizer boundary."""

    special_tokens = tokenizer_metadata.get("special_tokens", [])
    if not isinstance(special_tokens, list) or not all(isinstance(token, str) for token in special_tokens):
        raise ValueError("tokenizer metadata special_tokens must be a list of strings")
    if split_special_token not in special_tokens:
        raise ValueError(
            f"parallel splitting requires {split_special_token!r} to be registered as a tokenizer special token"
        )

    delimiter = split_special_token.encode("utf-8")
    for special_token in special_tokens:
        encoded = special_token.encode("utf-8")
        for split_offset in range(1, len(encoded)):
            suffix = encoded[split_offset:]
            comparison_size = min(len(suffix), len(delimiter))
            if suffix[:comparison_size] == delimiter[:comparison_size]:
                raise ValueError(
                    f"special token {special_token!r} can overlap a {split_special_token!r} split point; "
                    "choose an unambiguous split special token"
                )


def _iter_byte_range_text(
    input_path: Path,
    start: int,
    end: int,
    read_size: int,
) -> Iterator[str]:
    """Strictly decode one bounded byte range without reading it all at once."""

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    remaining = end - start
    with input_path.open("rb") as input_file:
        input_file.seek(start)
        while remaining:
            block = input_file.read(min(read_size, remaining))
            if not block:
                raise OSError(f"unexpected EOF while reading byte range [{start}, {end})")
            remaining -= len(block)
            text = decoder.decode(block, final=remaining == 0)
            if text:
                yield text
        if start == end:
            decoder.decode(b"", final=True)


def _write_token_ids(
    token_ids: Iterable[int],
    raw_path: Path,
    dtype: np.dtype,
    buffer_tokens: int,
) -> int:
    """Write an encoded stream to a headerless token part and return its length."""

    token_count = 0
    maximum_token_id = np.iinfo(dtype).max
    with raw_path.open("wb") as raw_file:
        buffer: list[int] = []
        for token_id in token_ids:
            if token_id < 0 or token_id > maximum_token_id:
                raise ValueError(f"encoded token ID {token_id} does not fit in {dtype.name}")
            buffer.append(token_id)
            if len(buffer) >= buffer_tokens:
                np.asarray(buffer, dtype=dtype).tofile(raw_file)
                token_count += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(raw_file)
            token_count += len(buffer)
        raw_file.flush()
        os.fsync(raw_file.fileno())
    return token_count


def _encode_byte_range(
    input_path: str,
    tokenizer_artifact: str,
    raw_path: str,
    start: int,
    end: int,
    dtype_name: str,
    chunk_chars: int,
    buffer_tokens: int,
) -> int:
    """Process-pool entry point: load a tokenizer and encode one byte range."""

    tokenizer, _, _ = load_tokenizer_artifact(tokenizer_artifact)
    chunks = _iter_byte_range_text(Path(input_path), start, end, chunk_chars)
    return _write_token_ids(tokenizer.encode_iterable(chunks), Path(raw_path), np.dtype(dtype_name), buffer_tokens)


def _new_raw_path(output: Path, part_index: int) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output.parent,
        prefix=f".{output.name}.part-{part_index:04d}.",
        suffix=".tokens.tmp",
        delete=False,
    )
    temporary.close()
    return Path(temporary.name)


def _encode_in_parallel(
    input_path: Path,
    tokenizer_artifact: Path,
    raw_paths: list[Path],
    boundaries: list[int],
    dtype: np.dtype,
    chunk_chars: int,
    buffer_tokens: int,
) -> list[int]:
    """Encode all byte ranges concurrently, returning counts in source order."""

    ranges = list(zip(boundaries, boundaries[1:]))
    token_counts = [0] * len(ranges)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(ranges), mp_context=context) as executor:
        future_indices = {
            executor.submit(
                _encode_byte_range,
                str(input_path),
                str(tokenizer_artifact),
                str(raw_paths[index]),
                start,
                end,
                dtype.name,
                chunk_chars,
                buffer_tokens,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        try:
            for future in as_completed(future_indices):
                index = future_indices[future]
                token_counts[index] = future.result()
        except BaseException:
            for future in future_indices:
                future.cancel()
            raise
    return token_counts


def _write_final_array(
    temporary_npy: Path,
    parts: list[tuple[Path, int]],
    dtype: np.dtype,
    copy_chunk_tokens: int,
) -> int:
    """Copy ordered raw parts into one mmap-compatible NumPy array."""

    token_count = sum(part_count for _, part_count in parts)
    output_array = np.lib.format.open_memmap(temporary_npy, mode="w+", dtype=dtype, shape=(token_count,))
    output_offset = 0
    for raw_path, part_count in parts:
        expected_size = part_count * dtype.itemsize
        if raw_path.stat().st_size != expected_size:
            raise RuntimeError(f"temporary token part {raw_path.name} has an unexpected size")
        if part_count:
            raw_array = np.memmap(raw_path, mode="r", dtype=dtype, shape=(part_count,))
            for start_index in range(0, part_count, copy_chunk_tokens):
                end_index = min(start_index + copy_chunk_tokens, part_count)
                output_array[output_offset + start_index : output_offset + end_index] = raw_array[start_index:end_index]
            del raw_array
        output_offset += part_count
    output_array.flush()
    del output_array
    with temporary_npy.open("rb") as output_file:
        os.fsync(output_file.fileno())
    return token_count


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    if args.chunk_chars <= 0 or args.buffer_tokens <= 0 or args.copy_chunk_tokens <= 0:
        raise ValueError("chunk and buffer sizes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.split_special_token:
        raise ValueError("--split-special-token cannot be empty")
    if args.workers > 1 and args.errors != "strict":
        raise ValueError("--workers greater than one requires --errors strict")
    sidecar = metadata_path_for(args.output)
    if not args.overwrite and (args.output.exists() or sidecar.exists()):
        raise FileExistsError("output already exists; pass --overwrite to replace it")

    tokenizer, tokenizer_metadata, tokenizer_metadata_path = load_tokenizer_artifact(args.tokenizer)
    maximum_token_id = max(tokenizer.vocab, default=-1)
    dtype = select_dtype(args.dtype, maximum_token_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_special_token_bytes = args.split_special_token.encode("utf-8")
    boundaries = find_chunk_boundaries(args.input, args.workers, split_special_token_bytes)
    effective_workers = len(boundaries) - 1
    if effective_workers > 1:
        _validate_parallel_split(tokenizer_metadata, args.split_special_token)

    raw_paths: list[Path] = []
    temporary_npy = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp.npy")
    start = time.perf_counter()
    token_count = 0
    try:
        for index in range(effective_workers):
            raw_paths.append(_new_raw_path(args.output, index))
        if effective_workers == 1:
            chunks = iter_text_chunks(args.input, args.chunk_chars, errors=args.errors)
            part_counts = [_write_token_ids(tokenizer.encode_iterable(chunks), raw_paths[0], dtype, args.buffer_tokens)]
        else:
            del tokenizer
            part_counts = _encode_in_parallel(
                args.input.resolve(),
                args.tokenizer.resolve(),
                raw_paths,
                boundaries,
                dtype,
                args.chunk_chars,
                args.buffer_tokens,
            )
        token_count = _write_final_array(
            temporary_npy,
            list(zip(raw_paths, part_counts, strict=True)),
            dtype,
            args.copy_chunk_tokens,
        )
        os.replace(temporary_npy, args.output)
    finally:
        temporary_npy.unlink(missing_ok=True)
        for raw_path in raw_paths:
            raw_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    source_bytes = args.input.stat().st_size
    summary = {
        "format": "cs336-token-array-v1",
        "created_at_utc": utc_timestamp(),
        "source_filename": args.input.name,
        "source_bytes": source_bytes,
        "tokenizer_artifact": tokenizer_metadata_path.name,
        "tokenizer_vocab_size": tokenizer_metadata["vocab_size"],
        "tokenizer_sha256": tokenizer_metadata.get("sha256"),
        "array_sha256": sha256_file(args.output),
        "token_count": token_count,
        "dtype": dtype.name,
        "shape": [token_count],
        "elapsed_seconds": elapsed,
        "bytes_per_token": source_bytes / token_count if token_count else None,
        "source_bytes_per_second": source_bytes / elapsed if elapsed else None,
        "requested_workers": args.workers,
        "effective_workers": effective_workers,
        "split_special_token": args.split_special_token,
    }
    atomic_write_json(sidecar, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
