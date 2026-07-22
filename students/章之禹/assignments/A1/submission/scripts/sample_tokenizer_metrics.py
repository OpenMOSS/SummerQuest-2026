#!/usr/bin/env python3
"""Measure tokenizer compression on a reproducible document reservoir sample."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def _sample_documents(
    input_path: Path,
    *,
    delimiter: str,
    sample_size: int,
    seed: int,
) -> tuple[list[tuple[int, str]], int]:
    """Return a uniform reservoir sample and the total document count."""

    if not delimiter:
        raise ValueError("document delimiter must not be empty")
    if sample_size <= 0:
        raise ValueError("sample size must be positive")

    generator = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    pending = ""
    document_count = 0

    def consider(document: str) -> None:
        nonlocal document_count
        document_index = document_count
        document_count += 1
        if len(reservoir) < sample_size:
            reservoir.append((document_index, document))
            return
        replacement = generator.randrange(document_count)
        if replacement < sample_size:
            reservoir[replacement] = (document_index, document)

    with input_path.open(encoding="utf-8") as corpus:
        for chunk in iter(lambda: corpus.read(1 << 20), ""):
            pending += chunk
            documents = pending.split(delimiter)
            pending = documents.pop()
            for document in documents:
                consider(document)
    if pending:
        consider(pending)

    reservoir.sort(key=lambda item: item[0])
    return reservoir, document_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--merges", required=True, type=Path)
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    parser.add_argument("--document-delimiter", default="<|endoftext|>")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    documents, population = _sample_documents(
        args.input_path,
        delimiter=args.document_delimiter,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        args.special_tokens or [],
    )

    started = time.perf_counter()
    byte_counts = [len(document.encode("utf-8")) for _, document in documents]
    token_counts = [len(tokenizer.encode(document)) for _, document in documents]
    elapsed = time.perf_counter() - started
    byte_count = sum(byte_counts)
    token_count = sum(token_counts)
    metrics = {
        "seed": args.seed,
        "document_delimiter": args.document_delimiter,
        "population_documents": population,
        "sampled_documents": len(documents),
        "sampled_document_indices": [index for index, _ in documents],
        "bytes": byte_count,
        "tokens": token_count,
        "bytes_per_token": byte_count / token_count if token_count else None,
        "bytes_per_second": byte_count / elapsed if elapsed else None,
        "elapsed_seconds": elapsed,
        "per_document_bytes_per_token": [
            byte_value / token_value if token_value else None
            for byte_value, token_value in zip(byte_counts, token_counts, strict=True)
        ],
    }
    rendered = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
