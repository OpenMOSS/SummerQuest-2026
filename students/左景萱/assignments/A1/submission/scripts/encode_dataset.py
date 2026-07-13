#!/usr/bin/env python3
"""Stream a UTF-8 corpus into a flat raw uint16/uint32 token file."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import BinaryIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import Tokenizer  # noqa: E402
from scripts.tokenizer_artifacts import (  # noqa: E402
    CorpusTextStream,
    resolve_special_tokens,
    utc_now,
    write_json_atomic,
)


_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_DTYPE: np.dtype | None = None
_WORKER_WRITE_BUFFER_TOKENS = 0


def _initialize_encode_worker(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    dtype_string: str,
    write_buffer_tokens: int,
) -> None:
    """Load immutable encoding state once in each spawned worker."""

    global _WORKER_DTYPE, _WORKER_TOKENIZER, _WORKER_WRITE_BUFFER_TOKENS
    _WORKER_TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    _WORKER_DTYPE = np.dtype(dtype_string)
    _WORKER_WRITE_BUFFER_TOKENS = write_buffer_tokens


def _encode_chunk_to_part(task: tuple[int, str, str]) -> tuple[int, str, int, int]:
    """Encode one stream chunk into an independent raw-token part file."""

    chunk_index, text, part_path_string = task
    if _WORKER_TOKENIZER is None or _WORKER_DTYPE is None:
        raise RuntimeError("encoding worker was not initialized")

    part_path = Path(part_path_string)
    token_buffer: list[int] = []
    token_count = 0
    with part_path.open("wb") as part_file:
        for token_id in _WORKER_TOKENIZER.encode_iterable((text,)):
            token_buffer.append(token_id)
            if len(token_buffer) >= _WORKER_WRITE_BUFFER_TOKENS:
                values = np.asarray(token_buffer, dtype=_WORKER_DTYPE)
                part_file.write(values.tobytes(order="C"))
                token_count += len(token_buffer)
                token_buffer.clear()
        if token_buffer:
            values = np.asarray(token_buffer, dtype=_WORKER_DTYPE)
            part_file.write(values.tobytes(order="C"))
            token_count += len(token_buffer)

    return chunk_index, part_path_string, token_count, part_path.stat().st_size


def _append_part(
    output_file: BinaryIO,
    part_path: Path,
    digest: hashlib._Hash,
    *,
    copy_buffer_bytes: int = 8 * 1024 * 1024,
) -> int:
    """Append a part to the final stream while updating its digest."""

    copied = 0
    with part_path.open("rb") as part_file:
        while raw := part_file.read(copy_buffer_bytes):
            output_file.write(raw)
            digest.update(raw)
            copied += len(raw)
    return copied


def _parallel_safe_chunks(stream: Iterable[str], document_delimiter: str) -> Iterator[str]:
    """Require every non-final worker chunk to end at a document boundary."""

    def ends_at_delimiter_match(text: str) -> bool:
        last_match_end = -1
        search_start = 0
        while (position := text.find(document_delimiter, search_start)) >= 0:
            last_match_end = position + len(document_delimiter)
            search_start = last_match_end
        return last_match_end == len(text)

    chunks = iter(stream)
    try:
        previous = next(chunks)
    except StopIteration:
        return

    for current in chunks:
        if not ends_at_delimiter_match(previous):
            raise ValueError(
                "parallel encoding encountered a non-final chunk without the document delimiter; "
                "rerun with --num-processes 1"
            )
        yield previous
        previous = current
    yield previous


def _encode_parallel(
    stream: Iterable[str],
    output_file: BinaryIO,
    digest: hashlib._Hash,
    *,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    dtype: np.dtype,
    write_buffer_tokens: int,
    num_processes: int,
    part_directory: Path,
) -> tuple[int, int]:
    """Encode stream chunks concurrently and append completed parts in order."""

    max_in_flight = max(1, 2 * num_processes)
    chunks = enumerate(stream)
    pending: dict[Future[tuple[int, str, int, int]], int] = {}
    completed: dict[int, tuple[Path, int, int]] = {}
    next_part_to_append = 0
    total_tokens = 0
    num_parts = 0
    chunks_exhausted = False

    executor = ProcessPoolExecutor(
        max_workers=num_processes,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_encode_worker,
        initargs=(
            os.fspath(vocab_path),
            os.fspath(merges_path),
            special_tokens,
            dtype.str,
            write_buffer_tokens,
        ),
    )
    try:
        while pending or not chunks_exhausted:
            while not chunks_exhausted and len(pending) + len(completed) < max_in_flight:
                try:
                    chunk_index, text = next(chunks)
                except StopIteration:
                    chunks_exhausted = True
                    break
                part_path = part_directory / f"{chunk_index:012d}.part"
                future = executor.submit(_encode_chunk_to_part, (chunk_index, text, os.fspath(part_path)))
                pending[future] = chunk_index
                num_parts += 1

            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                expected_index = pending.pop(future)
                chunk_index, part_path_string, part_tokens, part_bytes = future.result()
                if chunk_index != expected_index:
                    raise RuntimeError(f"worker returned part {chunk_index}, expected {expected_index}")
                completed[chunk_index] = (Path(part_path_string), part_tokens, part_bytes)

            while next_part_to_append in completed:
                part_path, part_tokens, expected_bytes = completed.pop(next_part_to_append)
                copied_bytes = _append_part(output_file, part_path, digest)
                if copied_bytes != expected_bytes or copied_bytes != part_tokens * dtype.itemsize:
                    raise RuntimeError(f"invalid encoded part: {part_path}")
                total_tokens += part_tokens
                part_path.unlink()
                next_part_to_append += 1
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    if completed or next_part_to_append != num_parts:
        raise RuntimeError("encoded parts were not appended completely")
    return total_tokens, num_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="UTF-8 corpus to encode.")
    parser.add_argument("--vocab", type=Path, required=True, help="GPT-2-style vocab.json.")
    parser.add_argument("--merges", type=Path, required=True, help="GPT-2-style merges.txt.")
    parser.add_argument("--output", type=Path, required=True, help="Raw .bin output path.")
    parser.add_argument("--metadata", type=Path, default=None, help="Default: OUTPUT.metadata.json.")
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help="Indivisible special token; repeat for multiple tokens. Default: <|endoftext|>.",
    )
    parser.add_argument("--no-special-tokens", action="store_true", help="Disable the default special token.")
    parser.add_argument("--dtype", choices=("auto", "uint16", "uint32"), default="auto")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--write-buffer-tokens", type=int, default=1_000_000)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Encoding workers. Parallel encoding requires exactly one document-boundary token.",
    )
    parser.add_argument("--max-bytes", type=int, default=None, help="Encode at most this many source bytes.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_dtype(requested: str, max_token_id: int) -> np.dtype:
    if max_token_id < 0:
        raise ValueError("tokenizer vocabulary is empty")
    if requested == "auto":
        requested = "uint16" if max_token_id <= np.iinfo(np.uint16).max else "uint32"
    dtype = np.dtype(requested)
    if max_token_id > np.iinfo(dtype).max:
        raise ValueError(f"token id {max_token_id} does not fit in {requested}")
    return dtype


def main() -> None:
    args = parse_args()
    if args.write_buffer_tokens < 1:
        raise ValueError("--write-buffer-tokens must be positive")
    if args.num_processes < 1:
        raise ValueError("--num-processes must be positive")
    special_tokens = resolve_special_tokens(args.special_tokens, args.no_special_tokens)
    if args.num_processes > 1 and len(special_tokens) != 1:
        raise ValueError("parallel encoding requires exactly one unambiguous document-boundary token")
    corpus = args.corpus.expanduser()
    vocab_path = args.vocab.expanduser()
    merges_path = args.merges.expanduser()
    output_path = args.output.expanduser()
    metadata_path = (
        args.metadata.expanduser()
        if args.metadata is not None
        else output_path.with_name(f"{output_path.name}.metadata.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == metadata_path.resolve():
        raise ValueError("token output and metadata paths must be distinct")
    if (output_path.exists() or metadata_path.exists()) and not args.overwrite:
        existing = output_path if output_path.exists() else metadata_path
        raise FileExistsError(f"output exists (pass --overwrite): {existing}")
    protected_inputs = {corpus.resolve(), vocab_path.resolve(), merges_path.resolve()}
    if output_path.resolve() in protected_inputs or metadata_path.resolve() in protected_inputs:
        raise ValueError("output and metadata paths must not overwrite an input")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    max_token_id = max(tokenizer.vocab)
    dtype = choose_dtype(args.dtype, max_token_id)
    delimiter = special_tokens[0] if special_tokens else None
    stream = CorpusTextStream(
        corpus,
        chunk_bytes=args.chunk_bytes,
        max_bytes=args.max_bytes,
        document_delimiter=delimiter,
    )

    file_descriptor, temporary_string = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.tmp-",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_string)
    part_directory: Path | None = None
    token_buffer: list[int] = []
    token_count = 0
    num_parts = 0
    digest = hashlib.sha256()
    start = time.perf_counter()
    try:
        with temporary.open("wb") as output_file:
            if args.num_processes > 1:
                if delimiter is None:
                    raise RuntimeError("parallel delimiter validation was not initialized")
                part_directory = Path(
                    tempfile.mkdtemp(
                        dir=output_path.parent,
                        prefix=f".{output_path.name}.parts-",
                    )
                )
                token_count, num_parts = _encode_parallel(
                    _parallel_safe_chunks(stream, delimiter),
                    output_file,
                    digest,
                    vocab_path=vocab_path,
                    merges_path=merges_path,
                    special_tokens=special_tokens,
                    dtype=dtype,
                    write_buffer_tokens=args.write_buffer_tokens,
                    num_processes=args.num_processes,
                    part_directory=part_directory,
                )
            else:
                for token_id in tokenizer.encode_iterable(stream):
                    token_buffer.append(token_id)
                    if len(token_buffer) < args.write_buffer_tokens:
                        continue
                    values = np.asarray(token_buffer, dtype=dtype)
                    raw = values.tobytes(order="C")
                    output_file.write(raw)
                    digest.update(raw)
                    token_count += len(token_buffer)
                    token_buffer.clear()
                if token_buffer:
                    values = np.asarray(token_buffer, dtype=dtype)
                    raw = values.tobytes(order="C")
                    output_file.write(raw)
                    digest.update(raw)
                    token_count += len(token_buffer)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if part_directory is not None:
            shutil.rmtree(part_directory, ignore_errors=True)
    elapsed = time.perf_counter() - start

    source = stream.summary()
    if args.num_processes == 1:
        num_parts = source["chunks"]
    processed_bytes = source["bytes_processed"]
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "source": source,
        "tokenizer": {
            "vocab": str(vocab_path),
            "merges": str(merges_path),
            "vocab_size": len(tokenizer.vocab),
            "max_token_id": max_token_id,
            "special_tokens": special_tokens,
        },
        "encoding": {
            "format": "raw_flat_token_ids",
            "dtype": dtype.name,
            "byte_order": sys.byteorder,
            "num_tokens": token_count,
            "output_bytes": output_path.stat().st_size,
            "sha256": digest.hexdigest(),
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else None,
            "source_bytes_per_second": processed_bytes / elapsed if elapsed > 0 else None,
            "compression_ratio_bytes_per_token": processed_bytes / token_count if token_count else None,
            "num_processes_requested": args.num_processes,
            "num_processes_used": 1 if args.num_processes == 1 else min(args.num_processes, num_parts),
            "num_parts": num_parts,
        },
        "output": {"tokens": str(output_path), "metadata": str(metadata_path)},
    }
    write_json_atomic(metadata_path, metadata)
    print(f"encoded {processed_bytes} bytes into {token_count} {dtype.name} tokens")
    print(f"saved {output_path} and {metadata_path}")


if __name__ == "__main__":
    main()
