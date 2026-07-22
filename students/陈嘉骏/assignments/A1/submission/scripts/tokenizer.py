from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from cs336_basics.tokenizer_experiments import (
    benchmark_tokenizer,
    encode_file_to_numpy_binary,
    load_tokenizer_artifact,
    longest_vocab_tokens,
    train_tokenizer_artifact,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, inspect, benchmark, and apply a byte-level BPE tokenizer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a BPE tokenizer artifact.")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--vocab-size", required=True, type=int)
    train_parser.add_argument("--special-token", action="append", default=[])
    train_parser.add_argument("--num-processes", type=int)
    train_parser.add_argument("--vocab-output")
    train_parser.add_argument("--merges-output")

    encode_parser = subparsers.add_parser("encode", help="Stream a corpus into a raw NumPy token ID file.")
    encode_parser.add_argument("--tokenizer", required=True)
    encode_parser.add_argument("--input", required=True)
    encode_parser.add_argument("--output", required=True)
    encode_parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    encode_parser.add_argument("--token-batch-size", type=int, default=65_536)
    encode_parser.add_argument("--dtype", choices=["uint16", "uint32", "uint64"])

    benchmark_parser = subparsers.add_parser("benchmark", help="Measure compression ratio and encode throughput.")
    benchmark_parser.add_argument("--tokenizer", required=True)
    benchmark_parser.add_argument("--input", required=True)
    benchmark_parser.add_argument("--chunk-size", type=int, default=1024 * 1024)

    inspect_parser = subparsers.add_parser("inspect", help="Show the longest tokens in a tokenizer vocabulary.")
    inspect_parser.add_argument("--tokenizer", required=True)
    inspect_parser.add_argument("--limit", type=int, default=10)
    return parser


def _run_train(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    tokenizer = train_tokenizer_artifact(
        input_path=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        num_processes=args.num_processes,
    )
    artifact_path = Path(args.output)
    vocab_output = Path(args.vocab_output) if args.vocab_output else artifact_path.with_suffix(".vocab.json")
    merges_output = Path(args.merges_output) if args.merges_output else artifact_path.with_suffix(".merges.json")
    tokenizer.to_files(vocab_output, merges_output)
    print(
        json.dumps(
            {
                "vocab_size": len(tokenizer.vocab),
                "num_merges": len(tokenizer.merges),
                "elapsed_seconds": time.perf_counter() - start_time,
                "output": os.fspath(args.output),
                "vocab_output": os.fspath(vocab_output),
                "merges_output": os.fspath(merges_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_encode(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer_artifact(args.tokenizer)
    info = encode_file_to_numpy_binary(
        tokenizer=tokenizer,
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        token_batch_size=args.token_batch_size,
        dtype=args.dtype,
    )
    print(json.dumps(asdict(info), ensure_ascii=False, indent=2))


def _run_benchmark(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer_artifact(args.tokenizer)
    benchmark = benchmark_tokenizer(tokenizer, args.input, chunk_size=args.chunk_size)
    print(json.dumps(asdict(benchmark), ensure_ascii=False, indent=2))


def _run_inspect(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer_artifact(args.tokenizer)
    tokens = [
        {
            "id": token_id,
            "num_bytes": len(token_bytes),
            "bytes_hex": token_bytes.hex(),
            "text": token_bytes.decode("utf-8", errors="replace"),
        }
        for token_id, token_bytes in longest_vocab_tokens(tokenizer, args.limit)
    ]
    print(json.dumps(tokens, ensure_ascii=False, indent=2))


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "train":
        _run_train(args)
    elif args.command == "encode":
        _run_encode(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "inspect":
        _run_inspect(args)


if __name__ == "__main__":
    main()
