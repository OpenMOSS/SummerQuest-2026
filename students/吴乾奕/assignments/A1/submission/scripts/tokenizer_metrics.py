#!/usr/bin/env python3
"""Measure tokenizer compression and throughput on one or more corpora."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, action="append")
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-documents", type=int, default=10)
    parser.add_argument("--document-separator", default="<|endoftext|>")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def measure(
    tokenizer: Tokenizer,
    path: Path,
    max_bytes: int,
    max_documents: int,
    document_separator: str,
) -> dict[str, object]:
    # Sampling a bounded prefix makes cross-tokenizer comparisons inexpensive
    # and repeatable while still exercising streaming encoding.
    consumed_bytes = 0
    token_count = 0
    document_count = 0

    def bounded_chunks():
        nonlocal consumed_bytes, document_count
        buffer = ""
        bytes_read = 0
        with path.open(encoding="utf-8", newline="") as input_file:
            while bytes_read < max_bytes and document_count < max_documents:
                chunk = input_file.read(1 << 20)
                if not chunk:
                    break
                encoded_chunk = chunk.encode("utf-8")
                remaining_budget = max_bytes - bytes_read
                reached_byte_limit = len(encoded_chunk) >= remaining_budget
                if reached_byte_limit:
                    encoded_chunk = encoded_chunk[:remaining_budget]
                    chunk = encoded_chunk.decode("utf-8", errors="ignore")
                    encoded_chunk = chunk.encode("utf-8")
                bytes_read += len(encoded_chunk)
                buffer += chunk
                while document_count < max_documents:
                    separator_index = buffer.find(document_separator)
                    if separator_index < 0:
                        break
                    end = separator_index + len(document_separator)
                    piece = buffer[:end]
                    piece_bytes = len(piece.encode("utf-8"))
                    if consumed_bytes + piece_bytes > max_bytes:
                        remaining = max_bytes - consumed_bytes
                        encoded = piece.encode("utf-8")[:remaining]
                        piece = encoded.decode("utf-8", errors="ignore")
                        if piece:
                            consumed_bytes += len(piece.encode("utf-8"))
                            yield piece
                        return
                    consumed_bytes += piece_bytes
                    document_count += 1
                    yield piece
                    buffer = buffer[end:]
                if reached_byte_limit:
                    break
            if buffer and document_count < max_documents:
                consumed_bytes += len(buffer.encode("utf-8"))
                document_count += 1
                yield buffer

    started = time.perf_counter()
    for _ in tokenizer.encode_iterable(bounded_chunks()):
        token_count += 1
    elapsed = time.perf_counter() - started
    return {
        "input": str(path),
        "sample_bytes": consumed_bytes,
        "document_count": document_count,
        "token_count": token_count,
        "bytes_per_token": consumed_bytes / token_count if token_count else None,
        "elapsed_seconds": elapsed,
        "bytes_per_second": consumed_bytes / elapsed if elapsed else None,
    }


def main() -> None:
    args = parse_args()
    if args.max_bytes <= 0 or args.max_documents <= 0:
        raise ValueError("--max-bytes and --max-documents must be positive")
    if not args.document_separator:
        raise ValueError("--document-separator cannot be empty")
    tokenizer = Tokenizer.from_directory(args.tokenizer_dir)
    results = {
        "tokenizer_dir": str(args.tokenizer_dir),
        "measurements": [
            measure(tokenizer, path, args.max_bytes, args.max_documents, args.document_separator) for path in args.input
        ],
    }
    rendered = json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
