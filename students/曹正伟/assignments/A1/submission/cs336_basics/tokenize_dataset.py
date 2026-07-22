"""Encode a text corpus into a memory-mappable NumPy token array."""

from __future__ import annotations

import argparse
import codecs
import json
import multiprocessing as mp
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from .pretokenization_example import (
    END_OF_TEXT,
    find_chunk_boundaries,
)

END_OF_TEXT_STRING = END_OF_TEXT.decode("ascii")
READ_SIZE = 1024 * 1024
WRITE_BUFFER_SIZE = 65_536

# These globals are initialized separately inside every multiprocessing worker.
# They are never shared between worker processes.
_worker_tokenizer: Tokenizer | None = None
_worker_dtype: np.dtype | None = None


@dataclass(frozen=True)
class ChunkTask:
    """Description of one half-open input byte range."""

    chunk_index: int
    start: int
    end: int
    input_path: Path
    temp_dir: Path


@dataclass(frozen=True)
class ChunkResult:
    """Information returned by one encoding worker."""

    chunk_index: int
    token_count: int
    input_bytes: int
    temp_path: Path


def _initialize_worker(
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    dtype_name: str,
) -> None:
    """Construct process-local worker state."""
    global _worker_tokenizer
    global _worker_dtype

    _worker_tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens,
    )

    _worker_dtype = np.dtype(dtype_name)


def _decoded_chunk(
    task: ChunkTask,
) -> Iterator[str]:
    """Decode exactly one byte range without reading an adjacent chunk."""
    remaining = task.end - task.start
    decoder = codecs.getincrementaldecoder("utf-8")("strict")

    with task.input_path.open("rb") as source:
        source.seek(task.start)

        while remaining:
            data = source.read(
                min(READ_SIZE, remaining)
            )

            if not data:
                raise EOFError(
                    f"Unexpected EOF in chunk {task.chunk_index}: "
                    f"{remaining} bytes remain"
                )

            remaining -= len(data)

            text = decoder.decode(
                data,
                final=False,
            )

            if text:
                yield text

        # This raises UnicodeDecodeError if an unsafe boundary was accidentally
        # placed in the middle of a UTF-8 code point.
        final_text = decoder.decode(
            b"",
            final=True,
        )

        if final_text:
            yield final_text


def _write_token_buffer(
    destination: object,
    buffer: list[int],
    dtype: np.dtype,
) -> int:
    """Write one token buffer as raw fixed-width integer data."""
    if not buffer:
        return 0

    values = np.asarray(
        buffer,
        dtype=dtype,
    )

    destination.write(
        values.tobytes(order="C")
    )

    count = len(buffer)
    buffer.clear()

    return count


def _encode_chunk(
    task: ChunkTask,
) -> ChunkResult:
    """Encode one input range and write its IDs to a private raw file."""
    if _worker_tokenizer is None or _worker_dtype is None:
        raise RuntimeError(
            "Worker was not initialized"
        )

    temp_path = (
        task.temp_dir
        / f"chunk-{task.chunk_index:08d}.tokens"
    )

    token_count = 0
    buffer: list[int] = []

    with temp_path.open("wb") as destination:
        for token_id in _worker_tokenizer.encode_iterable(
            _decoded_chunk(task)
        ):
            buffer.append(token_id)

            if len(buffer) >= WRITE_BUFFER_SIZE:
                token_count += _write_token_buffer(
                    destination,
                    buffer,
                    _worker_dtype,
                )

        token_count += _write_token_buffer(
            destination,
            buffer,
            _worker_dtype,
        )

    return ChunkResult(
        chunk_index=task.chunk_index,
        token_count=token_count,
        input_bytes=task.end - task.start,
        temp_path=temp_path,
    )


def _normalize_output_path(
    path: Path,
) -> Path:
    """Ensure the final output has a .npy suffix."""
    if path.suffix == ".npy":
        return path

    return Path(f"{path}.npy")


def _merge_chunks(
    output_path: Path,
    results: list[ChunkResult],
    dtype_name: str,
) -> int:
    """Create the final .npy once and merge raw chunk files in order."""
    dtype = np.dtype(dtype_name)

    token_count = sum(
        result.token_count
        for result in results
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # This is the only creation of the final standard .npy file. Worker files
    # contain raw integers and intentionally do not use the .npy suffix.
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(token_count,),
    )

    write_offset = 0

    try:
        for result in sorted(
            results,
            key=lambda item: item.chunk_index,
        ):
            expected_size = (
                result.token_count
                * dtype.itemsize
            )

            actual_size = result.temp_path.stat().st_size

            if actual_size != expected_size:
                raise ValueError(
                    f"Temporary chunk {result.chunk_index} has "
                    f"{actual_size} bytes; expected {expected_size}"
                )

            if result.token_count == 0:
                continue

            chunk_tokens = np.memmap(
                result.temp_path,
                mode="r",
                dtype=dtype,
                shape=(result.token_count,),
            )

            try:
                next_offset = (
                    write_offset
                    + result.token_count
                )

                output[
                    write_offset:next_offset
                ] = chunk_tokens

                write_offset = next_offset
            finally:
                del chunk_tokens

        if write_offset != token_count:
            raise RuntimeError(
                f"Merged {write_offset} tokens, "
                f"expected {token_count}"
            )

        output.flush()
    finally:
        # Release the file handle before TemporaryDirectory attempts cleanup,
        # which is especially important on Windows.
        del output

    return token_count


def _build_special_tokens(
    requested_special_tokens: list[str],
) -> list[str]:
    """Always include the delimiter used for safe chunk boundaries."""
    special_tokens = [END_OF_TEXT_STRING]
    seen = {END_OF_TEXT_STRING}

    for token in requested_special_tokens:
        if token not in seen:
            special_tokens.append(token)
            seen.add(token)

    return special_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "input_path",
        type=Path,
    )

    parser.add_argument(
        "output_path",
        type=Path,
    )

    parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--merges",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--dtype",
        choices=("uint16", "uint32"),
        default="uint16",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.workers <= 0:
        parser.error(
            "--workers must be greater than 0"
        )

    special_tokens = _build_special_tokens(
        args.special_token
    )

    # Load once in the main process only to validate the requested dtype.
    # Input text is not encoded here.
    validation_tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens,
    )

    if (
        args.dtype == "uint16"
        and max(
            validation_tokenizer.vocab,
            default=0,
        )
        > np.iinfo(np.uint16).max
    ):
        raise ValueError(
            "Vocabulary IDs do not fit in uint16; "
            "use --dtype uint32"
        )

    del validation_tokenizer

    started = time.perf_counter()

    with args.input_path.open("rb") as source:
        boundaries = find_chunk_boundaries(
            source,
            args.workers,
            END_OF_TEXT,
        )

    ranges = list(
        zip(
            boundaries,
            boundaries[1:],
        )
    )

    # find_chunk_boundaries returns [0] for an empty file. Preserve one empty
    # task so the program still creates a valid shape-(0,) .npy result.
    if not ranges:
        ranges = [(0, 0)]

    output_path = _normalize_output_path(
        args.output_path
    )

    # Placing temporary files beside the output is preferable when that
    # directory already exists. Otherwise the system temporary directory is
    # used until the final output directory is created during the merge.
    temporary_parent = (
        output_path.parent
        if output_path.parent.exists()
        else None
    )

    with tempfile.TemporaryDirectory(
        prefix="tokenize-dataset-",
        dir=temporary_parent,
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        tasks = [
            ChunkTask(
                chunk_index=chunk_index,
                start=start,
                end=end,
                input_path=args.input_path,
                temp_dir=temp_dir,
            )
            for chunk_index, (start, end)
            in enumerate(ranges)
        ]

        initializer_args = (
            args.vocab,
            args.merges,
            special_tokens,
            args.dtype,
        )

        if args.workers == 1:
            # Keep the one-worker path free of process startup overhead while
            # still constructing its own Tokenizer and its own pre-token cache.
            _initialize_worker(
                *initializer_args
            )

            results = [
                _encode_chunk(task)
                for task in tasks
            ]
        else:
            # Every child runs _initialize_worker and therefore constructs an
            # independent Tokenizer with an independent bounded LRU cache.
            with mp.Pool(
                processes=args.workers,
                initializer=_initialize_worker,
                initargs=initializer_args,
            ) as pool:
                # Completion order does not matter. chunk_index is returned and
                # the main process restores original file order before merging.
                results = list(
                    pool.imap_unordered(
                        _encode_chunk,
                        tasks,
                    )
                )

        results.sort(
            key=lambda result: result.chunk_index
        )

        token_count = _merge_chunks(
            output_path,
            results,
            args.dtype,
        )

    # TemporaryDirectory has now removed every worker file.
    elapsed = time.perf_counter() - started

    input_bytes = sum(
        result.input_bytes
        for result in results
    )

    actual_input_size = args.input_path.stat().st_size

    if input_bytes != actual_input_size:
        raise RuntimeError(
            f"Chunk ranges covered {input_bytes} input bytes, "
            f"but the file contains {actual_input_size} bytes"
        )

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(output_path),
        "workers": args.workers,
        "tokens": token_count,
        "input_bytes": input_bytes,
        "bytes_per_token": (
            input_bytes
            / max(token_count, 1)
        ),
        "elapsed_seconds": elapsed,
        "bytes_per_second": (
            input_bytes
            / max(elapsed, 1e-12)
        ),
        "dtype": args.dtype,
    }

    print(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()