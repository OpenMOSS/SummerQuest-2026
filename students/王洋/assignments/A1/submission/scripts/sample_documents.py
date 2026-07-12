#!/usr/bin/env python3
"""Extract a deterministic prefix of delimiter-separated documents."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the first N documents from a text corpus without loading it all into memory."
    )
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 source corpus")
    parser.add_argument("--output", type=Path, required=True, help="UTF-8 sample destination")
    parser.add_argument("--count", type=int, default=10, help="number of documents to retain")
    parser.add_argument("--separator", default="<|endoftext|>", help="document delimiter")
    parser.add_argument("--chunk-chars", type=int, default=1024 * 1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.count <= 0 or args.chunk_chars <= 0:
        raise ValueError("count and chunk-chars must be positive")
    if not args.separator:
        raise ValueError("separator must not be empty")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    documents = 0
    carry = ""
    try:
        with args.input.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as destination:
            while documents < args.count:
                chunk = source.read(args.chunk_chars)
                if not chunk:
                    break
                carry += chunk
                cursor = 0
                while documents < args.count:
                    boundary = carry.find(args.separator, cursor)
                    if boundary < 0:
                        break
                    end = boundary + len(args.separator)
                    destination.write(carry[cursor:end])
                    cursor = end
                    documents += 1
                carry = carry[cursor:]
                if documents < args.count and len(carry) > len(args.separator):
                    flush_length = len(carry) - len(args.separator) + 1
                    destination.write(carry[:flush_length])
                    carry = carry[flush_length:]

            if documents < args.count:
                destination.write(carry)
            destination.flush()
            os.fsync(destination.fileno())

        if documents != args.count:
            raise ValueError(f"source contains only {documents} complete documents, fewer than requested {args.count}")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"wrote {documents} documents to {args.output}")


if __name__ == "__main__":
    main()
