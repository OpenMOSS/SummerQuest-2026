"""Utilities for splitting text corpora into independently tokenizable ranges.

The tokenizer scripts intentionally encode one text line at a time.  Splitting only
at line boundaries therefore preserves the exact output of serial
``BPETokenizer.encode_iterable(file)`` while allowing ranges to be processed by
different processes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from cs336_basics.bpe import BPETokenizer


def enable_encode_cache(tokenizer: BPETokenizer, max_size: int = 65_536) -> None:
    """Add a bounded, instance-local cache around BPE pre-token encoding.

    This is opt-in because the assignment's base tokenizer is also tested under
    a very small memory limit.  Long-running dataset scripts benefit greatly
    because common words and whitespace pre-tokens recur many times.
    """
    if max_size < 0:
        raise ValueError("max_size cannot be negative")
    if max_size == 0:
        return
    cached_encode_word = lru_cache(maxsize=max_size)(tokenizer._encode_word)
    setattr(tokenizer, "_encode_word", cached_encode_word)


def _find_next_token(file: BinaryIO, offset: int, end: int, token: bytes) -> int:
    """Find the next token start without missing matches across read chunks."""
    file.seek(offset)
    overlap = b""
    position = offset
    read_size = 64 * 1024

    while position < end:
        chunk = file.read(min(read_size, end - position))
        if not chunk:
            break
        combined = overlap + chunk
        found_at = combined.find(token)
        if found_at != -1:
            return position - len(overlap) + found_at
        overlap = combined[-(len(token) - 1) :] if len(token) > 1 else b""
        position += len(chunk)
    return end


def find_special_token_aligned_ranges(
    input_path: Path,
    desired_num_ranges: int,
    special_token: str,
    max_bytes: int | None = None,
) -> list[tuple[int, int]]:
    """Split a corpus at special-token starts, which are safe BPE boundaries.

    If ``max_bytes`` is supplied, the sampled prefix is extended to the next
    special-token boundary so a document is never cut in the middle.
    """
    if desired_num_ranges <= 0:
        raise ValueError("desired_num_ranges must be positive")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive when provided")
    if not special_token:
        raise ValueError("special_token cannot be empty")

    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return []

    token_bytes = special_token.encode("utf-8")
    with input_path.open("rb") as file:
        if max_bytes is None or max_bytes >= file_size:
            end = file_size
        else:
            end = _find_next_token(file, max_bytes, file_size, token_bytes)

        boundaries = [0]
        for index in range(1, desired_num_ranges):
            target = end * index // desired_num_ranges
            boundaries.append(_find_next_token(file, target, end, token_bytes))
        boundaries.append(end)

    unique_boundaries = sorted(set(boundaries))
    return [
        (start, stop)
        for start, stop in zip(unique_boundaries, unique_boundaries[1:])
        if start < stop
    ]


def iter_tokenizable_segments_in_range(
    input_path: Path,
    start: int,
    end: int,
    special_token: str,
) -> Iterator[str]:
    """Yield complete documents and special tokens from an aligned byte range.

    Each yielded segment can be encoded independently without changing the token
    sequence. Memory usage is bounded by the largest document rather than range
    size. Newlines are normalized like ordinary text-file reading.
    """
    if start < 0 or end < start:
        raise ValueError(f"invalid byte range [{start}, {end})")
    if not special_token:
        raise ValueError("special_token cannot be empty")

    token_bytes = special_token.encode("utf-8")
    buffer = b""
    with input_path.open("rb") as file:
        file.seek(start)
        while file.tell() < end:
            chunk = file.read(min(1024 * 1024, end - file.tell()))
            if not chunk:
                break
            buffer += chunk

            while (token_at := buffer.find(token_bytes)) != -1:
                document = buffer[:token_at]
                if document:
                    yield document.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                yield special_token
                buffer = buffer[token_at + len(token_bytes) :]

    if buffer:
        yield buffer.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
