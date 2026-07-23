"""Find safe byte ranges for parallel corpus pre-tokenization."""

from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Iterator
from typing import BinaryIO


END_OF_TEXT = b"<|endoftext|>"


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Return ordered byte offsets aligned to a special-token delimiter.

    Every interior boundary is placed at the first occurrence of
    ``split_special_token`` at or after its approximately evenly spaced target.

    With the ASCII delimiter ``b"<|endoftext|>"``:

    - a UTF-8 code point is never split;
    - the special token itself is never split;
    - the delimiter belongs entirely to the chunk on its right.

    Fewer than ``desired_num_chunks`` ranges may be returned when multiple
    candidate boundaries resolve to the same delimiter, or when no additional
    delimiter can be found.
    """
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")

    if not isinstance(split_special_token, bytes) or not split_special_token:
        raise ValueError("split_special_token must be a non-empty bytestring")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]

    chunk_size = max(1, file_size // desired_num_chunks)

    boundaries = [
        i * chunk_size
        for i in range(desired_num_chunks + 1)
    ]
    boundaries[-1] = file_size

    mini_chunk_size = 4096
    overlap = len(split_special_token) - 1

    for boundary_index in range(1, len(boundaries) - 1):
        search_position = boundaries[boundary_index]

        while search_position < file_size:
            file.seek(search_position)

            # The overlap ensures that a delimiter beginning near the end of
            # one 4 KiB block is still visible in this search block.
            mini_chunk = file.read(mini_chunk_size + overlap)

            if not mini_chunk:
                boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                boundaries[boundary_index] = search_position + found_at
                break

            search_position += mini_chunk_size
        else:
            boundaries[boundary_index] = file_size

    return sorted(set(boundaries))


def chunk_ranges(
    boundaries: list[int],
) -> Iterator[tuple[int, int]]:
    """Yield half-open byte ranges in their original file order."""
    yield from zip(boundaries, boundaries[1:])


def main() -> None:
    """Print safe pre-tokenization chunk boundaries for a corpus file."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("--chunks", type=int, default=4)
    args = parser.parse_args()

    with open(args.input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            args.chunks,
            END_OF_TEXT,
        )

    print(boundaries)


if __name__ == "__main__":
    main()