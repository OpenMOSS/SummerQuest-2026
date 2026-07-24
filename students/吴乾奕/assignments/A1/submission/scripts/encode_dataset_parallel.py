#!/usr/bin/env python3
"""Encode a corpus in exact, document-boundary-aligned parallel shards.

The regular :mod:`scripts.encode_dataset` entry point is intentionally a
single streaming process.  This helper accelerates large corpora without
changing tokenization semantics: it cuts the input only immediately after a
tokenizer special token, encodes each regular-file shard in a separate Python
process, and concatenates the resulting ``.npy`` arrays in input order.

Arbitrary byte or line cuts are not safe for this tokenizer because a
pre-token can span a chunk boundary.  A special token is a hard boundary, so
ending a shard immediately after one preserves the exact token stream.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=["auto", "uint16", "uint32", "int64"], default="auto")
    parser.add_argument("--workers", type=int, default=4, help="maximum number of encoding processes")
    parser.add_argument("--buffer-tokens", type=int, default=1_000_000)
    parser.add_argument(
        "--document-separator",
        default="<|endoftext|>",
        help="preferred special token at which shards may end; all tokenizer special tokens are also safe boundaries",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="keep temporary text and token shards after a successful merge",
    )
    return parser.parse_args()


def _choose_dtype(requested: str, max_token_id: int) -> np.dtype:
    if requested == "auto":
        return np.dtype(np.uint16 if max_token_id <= np.iinfo(np.uint16).max else np.uint32)
    dtype = np.dtype(requested)
    if np.issubdtype(dtype, np.unsignedinteger) and max_token_id > np.iinfo(dtype).max:
        raise ValueError(f"token ID {max_token_id} does not fit in {dtype}")
    return dtype


def _boundary_tokens(tokenizer: Tokenizer, preferred: str) -> tuple[bytes, ...]:
    configured = tuple(token.encode("utf-8") for token in tokenizer.special_tokens)
    preferred_bytes = preferred.encode("utf-8")
    if preferred_bytes not in configured:
        raise ValueError(
            f"--document-separator {preferred!r} is not present in tokenizer special_tokens "
            f"{list(tokenizer.special_tokens)!r}"
        )
    # Include every configured special token.  This remains exact if a future
    # tokenizer uses more than the assignment's usual end-of-text token.
    return tuple(sorted(set(configured), key=lambda token: (-len(token), token)))


def _next_boundary(mm: mmap.mmap, start: int, tokens: tuple[bytes, ...]) -> tuple[int, int] | None:
    """Return ``(position, length)`` of the first special token at/after start."""

    best_position: int | None = None
    best_length = 0
    for token in tokens:
        position = mm.find(token, start)
        if position < 0:
            continue
        if best_position is None or position < best_position:
            best_position = position
            best_length = len(token)
        elif position == best_position:
            best_length = max(best_length, len(token))
    if best_position is None:
        return None
    return best_position, best_length


def _copy_range(mm: mmap.mmap, start: int, end: int, destination: Path) -> int:
    copied = 0
    with destination.open("wb") as output_file:
        position = start
        while position < end:
            chunk_end = min(end, position + 64 * 1024 * 1024)
            output_file.write(mm[position:chunk_end])
            copied += chunk_end - position
            position = chunk_end
    return copied


def _make_shards(
    input_path: Path,
    shard_dir: Path,
    requested_shards: int,
    boundary_tokens: tuple[bytes, ...],
) -> list[Path]:
    """Create regular-file shards whose ends are special-token boundaries."""

    file_size = input_path.stat().st_size
    shard_dir.mkdir(parents=True, exist_ok=True)
    if file_size == 0:
        empty_path = shard_dir / "shard-0000.txt"
        empty_path.touch()
        return [empty_path]

    if not boundary_tokens:
        requested_shards = 1

    # The mmap is read-only and closed before workers are forked.  Thus the
    # parent never loads the corpus into Python heap memory and workers only
    # see their own regular-file shard.
    with input_path.open("rb") as input_file:
        with mmap.mmap(input_file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            starts = [0]
            ends: list[int] = []
            for index in range(1, requested_shards):
                target = file_size * index // requested_shards
                boundary = _next_boundary(mapped, target, boundary_tokens)
                if boundary is None:
                    break
                position, token_length = boundary
                end = position + token_length
                # A very long document or overlapping boundary can make the
                # next candidate precede the current start.  Skip it rather
                # than creating an empty or overlapping shard.
                if end <= starts[-1] or end >= file_size:
                    continue
                ends.append(end)
                starts.append(end)
            ends.append(file_size)

            shard_paths: list[Path] = []
            copied_total = 0
            for shard_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                shard_path = shard_dir / f"shard-{shard_index:04d}.txt"
                copied_total += _copy_range(mapped, start, end, shard_path)
                shard_paths.append(shard_path)
            if copied_total != file_size:
                raise OSError(f"shard byte total {copied_total} does not equal input size {file_size}")
    return shard_paths


def _encode_one_shard(
    input_path: str,
    tokenizer_dir: str,
    output_path: str,
    dtype_name: str,
    buffer_tokens: int,
) -> dict[str, Any]:
    """Worker implementation matching ``scripts/encode_dataset.py``."""

    source = Path(input_path)
    output_path_obj = Path(output_path)
    dtype = np.dtype(dtype_name)
    tokenizer = Tokenizer.from_directory(tokenizer_dir)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    token_count = 0
    bytes_read = source.stat().st_size
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=output_path_obj.name + ".",
            suffix=".tokens.tmp",
            dir=output_path_obj.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            buffer: list[int] = []
            with source.open(encoding="utf-8", newline="") as input_file:
                for token_id in tokenizer.encode_iterable(input_file):
                    buffer.append(token_id)
                    if len(buffer) >= buffer_tokens:
                        np.asarray(buffer, dtype=dtype).tofile(temporary_file)
                        token_count += len(buffer)
                        buffer.clear()
            if buffer:
                np.asarray(buffer, dtype=dtype).tofile(temporary_file)
                token_count += len(buffer)

        output = np.lib.format.open_memmap(output_path_obj, mode="w+", dtype=dtype, shape=(token_count,))
        copied = 0
        with temporary_path.open("rb") as temporary_file:
            while copied < token_count:
                count = min(buffer_tokens, token_count - copied)
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
        "input": str(source),
        "tokenizer_dir": str(tokenizer_dir),
        "output": str(output_path_obj),
        "dtype": dtype.name,
        "token_count": token_count,
        "input_bytes": bytes_read,
        "bytes_per_token": bytes_read / token_count if token_count else None,
        "elapsed_seconds": elapsed,
        "input_bytes_per_second": bytes_read / elapsed if elapsed else None,
    }
    report_path = output_path_obj.with_suffix(output_path_obj.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _merge_shards(
    shard_reports: list[dict[str, Any]],
    output_path: Path,
    tokenizer_dir: Path,
    input_path: Path,
    dtype: np.dtype,
    buffer_tokens: int,
    started: float,
    workers_requested: int,
    boundary_tokens: tuple[bytes, ...],
) -> dict[str, Any]:
    token_count = sum(int(report["token_count"]) for report in shard_reports)
    input_bytes = sum(int(report["input_bytes"]) for report in shard_reports)
    if input_bytes != input_path.stat().st_size:
        raise OSError(f"shard input bytes {input_bytes} do not equal source size {input_path.stat().st_size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
    partial_report_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.report.json.partial")
    try:
        output = np.lib.format.open_memmap(partial_path, mode="w+", dtype=dtype, shape=(token_count,))
        offset = 0
        for report in shard_reports:
            shard_array = np.load(report["output"], mmap_mode="r", allow_pickle=False)
            if shard_array.ndim != 1 or shard_array.dtype != dtype:
                raise ValueError(
                    f"unexpected shard array {report['output']}: shape={shard_array.shape}, dtype={shard_array.dtype}"
                )
            if len(shard_array) != int(report["token_count"]):
                raise OSError(f"shard report count disagrees with array: {report['output']}")
            copied = 0
            while copied < len(shard_array):
                count = min(buffer_tokens, len(shard_array) - copied)
                output[offset + copied : offset + copied + count] = shard_array[copied : copied + count]
                copied += count
            offset += len(shard_array)
            del shard_array
        if offset != token_count:
            raise OSError(f"merged {offset} tokens but expected {token_count}")
        output.flush()
        del output

        elapsed = time.perf_counter() - started
        report = {
            "input": str(input_path),
            "tokenizer_dir": str(tokenizer_dir),
            "output": str(output_path),
            "dtype": dtype.name,
            "token_count": token_count,
            "input_bytes": input_bytes,
            "bytes_per_token": input_bytes / token_count if token_count else None,
            "elapsed_seconds": elapsed,
            "input_bytes_per_second": input_bytes / elapsed if elapsed else None,
            "parallel": {
                "workers_requested": workers_requested,
                "workers_used": len(shard_reports),
                "shard_count": len(shard_reports),
                "boundary_tokens": [token.decode("utf-8", errors="replace") for token in boundary_tokens],
                "exact_document_boundaries": True,
            },
        }
        partial_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial_path, output_path)
        os.replace(partial_report_path, output_path.with_suffix(output_path.suffix + ".report.json"))
        return report
    finally:
        partial_path.unlink(missing_ok=True)
        partial_report_path.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.buffer_tokens <= 0:
        raise ValueError("--buffer-tokens must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.tokenizer_dir.is_dir():
        raise FileNotFoundError(args.tokenizer_dir)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --input")

    tokenizer = Tokenizer.from_directory(args.tokenizer_dir)
    dtype = _choose_dtype(args.dtype, max(tokenizer.vocab))
    boundary_tokens = _boundary_tokens(tokenizer, args.document_separator)
    worker_count = min(args.workers, max(1, os.cpu_count() or 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.parallel-", dir=args.output.parent or Path(".")))
    started = time.perf_counter()
    success = False
    try:
        shard_paths = _make_shards(args.input, work_dir / "text", worker_count, boundary_tokens)
        tasks = [
            (
                str(shard_path),
                str(args.tokenizer_dir),
                str(work_dir / "tokens" / f"shard-{index:04d}.npy"),
                dtype.name,
                args.buffer_tokens,
            )
            for index, shard_path in enumerate(shard_paths)
        ]
        print(
            f"created {len(tasks)} exact-boundary shards; starting up to {min(worker_count, len(tasks))} workers",
            file=sys.stderr,
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=min(worker_count, len(tasks))) as executor:
            futures = {executor.submit(_encode_one_shard, *task): index for index, task in enumerate(tasks)}
            indexed_reports: dict[int, dict[str, Any]] = {}
            for completed_count, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                indexed_reports[index] = future.result()
                print(
                    f"finished shard {completed_count}/{len(tasks)} (input shard {index + 1:04d})",
                    file=sys.stderr,
                    flush=True,
                )
        reports = [indexed_reports[index] for index in range(len(tasks))]
        report = _merge_shards(
            reports,
            args.output,
            args.tokenizer_dir,
            args.input,
            dtype,
            args.buffer_tokens,
            started,
            args.workers,
            boundary_tokens,
        )
        success = True
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    finally:
        if success and args.keep_work_dir:
            print(f"kept parallel work directory: {work_dir}", file=sys.stderr)
        elif success:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"parallel encoding failed; temporary files kept at: {work_dir}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
