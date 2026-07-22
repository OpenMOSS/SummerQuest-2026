#!/usr/bin/env python3
"""Measure tokenizer compression, longest token, and encoding throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import Tokenizer  # noqa: E402
from scripts.tokenizer_artifacts import (  # noqa: E402
    CorpusTextStream,
    longest_token_summary,
    resolve_special_tokens,
    utc_now,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="UTF-8 corpus used for measurement.")
    parser.add_argument("--vocab", type=Path, required=True, help="GPT-2-style vocab.json.")
    parser.add_argument("--merges", type=Path, required=True, help="GPT-2-style merges.txt.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics path.")
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help="Indivisible special token; repeat for multiple tokens. Default: <|endoftext|>.",
    )
    parser.add_argument("--no-special-tokens", action="store_true", help="Disable the default special token.")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-bytes", type=int, default=None, help="Measure at most this many source bytes.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = resolve_special_tokens(args.special_tokens, args.no_special_tokens)
    corpus = args.corpus.expanduser()
    vocab_path = args.vocab.expanduser()
    merges_path = args.merges.expanduser()
    output_path = args.output.expanduser() if args.output is not None else None
    if output_path is not None and output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")
    if output_path is not None and output_path.resolve() in {
        corpus.resolve(),
        vocab_path.resolve(),
        merges_path.resolve(),
    }:
        raise ValueError("metrics output must not overwrite an input")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    stream = CorpusTextStream(
        corpus,
        chunk_bytes=args.chunk_bytes,
        max_bytes=args.max_bytes,
        document_delimiter=special_tokens[0] if special_tokens else None,
    )
    token_count = 0
    start = time.perf_counter()
    for _ in tokenizer.encode_iterable(stream):
        token_count += 1
    elapsed = time.perf_counter() - start
    source = stream.summary()
    processed_bytes = source["bytes_processed"]

    metrics = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "source": source,
        "tokenizer": {
            "vocab": str(vocab_path),
            "merges": str(merges_path),
            "vocab_size": len(tokenizer.vocab),
            "special_tokens": special_tokens,
            "longest_token": longest_token_summary(tokenizer.vocab, special_tokens),
        },
        "metrics": {
            "num_tokens": token_count,
            "compression_ratio_bytes_per_token": processed_bytes / token_count if token_count else None,
            "elapsed_seconds": elapsed,
            "source_bytes_per_second": processed_bytes / elapsed if elapsed > 0 else None,
            "source_megabytes_per_second": processed_bytes / 1_000_000 / elapsed if elapsed > 0 else None,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else None,
        },
    }
    if output_path is not None:
        write_json_atomic(output_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
