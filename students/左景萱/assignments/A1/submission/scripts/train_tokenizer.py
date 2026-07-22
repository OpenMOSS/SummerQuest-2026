#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer and save GPT-2-compatible artifacts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import Tokenizer, train_bpe  # noqa: E402
from scripts.tokenizer_artifacts import (  # noqa: E402
    longest_token_summary,
    resolve_special_tokens,
    utc_now,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="UTF-8 training corpus.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size including special tokens.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help="Indivisible special token; repeat for multiple tokens. Default: <|endoftext|>.",
    )
    parser.add_argument("--no-special-tokens", action="store_true", help="Disable the default special token.")
    parser.add_argument("--num-processes", type=int, default=1, help="Workers used for corpus pre-token counting.")
    parser.add_argument("--vocab-filename", default="vocab.json")
    parser.add_argument("--merges-filename", default="merges.txt")
    parser.add_argument("--summary-filename", default="tokenizer_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = resolve_special_tokens(args.special_tokens, args.no_special_tokens)
    corpus = args.corpus.expanduser()
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    if args.num_processes < 1:
        raise ValueError("--num-processes must be positive")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / args.vocab_filename
    merges_path = output_dir / args.merges_filename
    summary_path = output_dir / args.summary_filename
    outputs = (vocab_path, merges_path, summary_path)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("vocab, merges, and summary paths must be distinct")
    if corpus.resolve() in {path.resolve() for path in outputs}:
        raise ValueError("tokenizer artifacts must not overwrite the input corpus")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {existing[0]}")

    started_at = utc_now()
    start = time.perf_counter()
    vocab, merges = train_bpe(
        corpus,
        args.vocab_size,
        special_tokens,
        num_processes=args.num_processes,
    )
    elapsed = time.perf_counter() - start
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    vocab_temporary = vocab_path.with_name(f"{vocab_path.name}.tmp")
    merges_temporary = merges_path.with_name(f"{merges_path.name}.tmp")
    tokenizer.save(vocab_temporary, merges_temporary)
    vocab_temporary.replace(vocab_path)
    merges_temporary.replace(merges_path)

    corpus_bytes = corpus.stat().st_size
    summary = {
        "schema_version": 1,
        "artifact_format": "gpt2_byte_level_bpe",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "corpus": {"path": str(corpus), "bytes": corpus_bytes},
        "configuration": {
            "vocab_size_requested": args.vocab_size,
            "special_tokens": special_tokens,
            "num_processes": args.num_processes,
        },
        "result": {
            "vocab_size": len(tokenizer.vocab),
            "num_merges": len(tokenizer.merges),
            "elapsed_seconds": elapsed,
            "corpus_bytes_per_second": corpus_bytes / elapsed if elapsed > 0 else None,
            "longest_token": longest_token_summary(tokenizer.vocab, special_tokens),
        },
        "artifacts": {
            "vocab": str(vocab_path),
            "merges": str(merges_path),
            "summary": str(summary_path),
        },
    }
    write_json_atomic(summary_path, summary)
    print(f"saved tokenizer ({len(tokenizer.vocab)} tokens, {len(merges)} merges) to {output_dir}")
    print(f"training time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
