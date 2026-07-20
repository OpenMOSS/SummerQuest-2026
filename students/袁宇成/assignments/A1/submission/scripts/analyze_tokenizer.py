#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import Tokenizer


def read_document_sample(path: Path, num_documents: int, max_bytes: int) -> bytes:
    delimiter = b"<|endoftext|>"
    data = bytearray()
    with path.open("rb") as file:
        while len(data) < max_bytes and data.count(delimiter) < num_documents:
            chunk = file.read(min(1 << 20, max_bytes - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    if num_documents > 0:
        end = 0
        for _ in range(num_documents):
            found = data.find(delimiter, end)
            if found < 0:
                break
            end = found + len(delimiter)
        if end:
            del data[end:]
    return bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tokenizer compression and throughput on a text sample.")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--num-documents", type=int, default=10)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    raw = read_document_sample(Path(args.text), args.num_documents, args.max_bytes)
    text = raw.decode("utf-8", errors="ignore")
    raw = text.encode("utf-8")
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    longest_id, longest_bytes = max(tokenizer.vocab.items(), key=lambda item: len(item[1]))
    result = {
        "text": Path(args.text).name,
        "documents": min(args.num_documents, raw.count(b"<|endoftext|>")),
        "bytes": len(raw),
        "tokens": len(ids),
        "compression_bytes_per_token": len(raw) / len(ids),
        "throughput_bytes_per_sec": len(raw) / elapsed,
        "throughput_tokens_per_sec": len(ids) / elapsed,
        "longest_token_id": longest_id,
        "longest_token_num_bytes": len(longest_bytes),
        "longest_token_repr": longest_bytes.decode("utf-8", errors="replace"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
