from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import BinaryIO

from cs336_basics.tokenizer import Tokenizer


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must use NAME=PATH") from error
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("must use non-empty NAME=PATH")
    return name, Path(raw_path)


def find_marker(source: BinaryIO, start: int, marker: bytes, file_size: int) -> int | None:
    source.seek(start)
    overlap = b""
    position = start
    overlap_size = max(0, len(marker) - 1)
    while position < file_size:
        block = source.read(min(1024 * 1024, file_size - position))
        if not block:
            return None
        searchable = overlap + block
        found_at = searchable.find(marker)
        if found_at >= 0:
            return position - len(overlap) + found_at
        overlap = searchable[-overlap_size:] if overlap_size else b""
        position += len(block)
    return None


def sample_documents(input_path: Path, count: int, delimiter: bytes) -> list[tuple[int, str]]:
    """Read deterministic documents spread approximately evenly across a corpus."""
    file_size = input_path.stat().st_size
    samples: list[tuple[int, str]] = []
    seen_starts: set[int] = set()

    with input_path.open("rb") as source:
        for sample_index in range(count):
            target = file_size * sample_index // count
            if target == 0:
                document_start = 0
            else:
                preceding_document_end = find_marker(source, target, delimiter, file_size)
                if preceding_document_end is None:
                    continue
                document_start = preceding_document_end + len(delimiter)

            if document_start in seen_starts or document_start >= file_size:
                continue
            document_end = find_marker(source, document_start, delimiter, file_size)
            if document_end is None:
                document_end = file_size

            source.seek(document_start)
            document_bytes = source.read(document_end - document_start)
            samples.append((document_start, document_bytes.decode("utf-8")))
            seen_starts.add(document_start)

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tokenizer compression on deterministic corpus documents and extrapolate throughput."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=parse_named_path,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Tokenizer label and JSON path; repeat to compare tokenizers.",
    )
    parser.add_argument("--documents", type=positive_int, default=10)
    parser.add_argument("--delimiter", default="<|endoftext|>")
    parser.add_argument("--cache-size", type=positive_int, default=32768)
    parser.add_argument(
        "--throughput-json",
        type=parse_named_path,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional benchmark JSON; repeat to extrapolate each measured throughput.",
    )
    parser.add_argument("--target-gb", type=float, default=825.0, help="Decimal GB used for throughput extrapolation.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_gb <= 0:
        raise ValueError("target-gb must be positive")

    documents = sample_documents(args.input, args.documents, args.delimiter.encode("utf-8"))
    if len(documents) != args.documents:
        raise RuntimeError(f"requested {args.documents} documents but found only {len(documents)}")

    tokenizer_results: dict[str, object] = {}
    for name, tokenizer_path in args.tokenizer:
        tokenizer = Tokenizer.load(tokenizer_path)
        tokenizer.enable_merge_cache(args.cache_size)
        per_document: list[dict[str, int | float]] = []
        total_bytes = 0
        total_tokens = 0
        started = time.perf_counter()
        for document_index, (byte_offset, document) in enumerate(documents):
            byte_count = len(document.encode("utf-8"))
            token_count = len(tokenizer.encode(document))
            total_bytes += byte_count
            total_tokens += token_count
            per_document.append(
                {
                    "document_index": document_index,
                    "byte_offset": byte_offset,
                    "bytes": byte_count,
                    "tokens": token_count,
                    "bytes_per_token": byte_count / token_count if token_count else 0.0,
                }
            )
        elapsed = time.perf_counter() - started
        tokenizer_results[name] = {
            "tokenizer_name": tokenizer_path.name,
            "documents": per_document,
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "bytes_per_token": total_bytes / total_tokens if total_tokens else None,
            "sample_elapsed_sec": elapsed,
            "sample_tokens_per_sec": total_tokens / elapsed if elapsed else None,
        }

    target_bytes = args.target_gb * 1_000_000_000
    extrapolations: dict[str, object] = {}
    for name, metrics_path in args.throughput_json:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        bytes_per_second = None
        if metrics.get("megabytes_per_sec") is not None:
            bytes_per_second = float(metrics["megabytes_per_sec"]) * 1_000_000
        elif metrics.get("bytes") is not None and metrics.get("elapsed_sec"):
            bytes_per_second = float(metrics["bytes"]) / float(metrics["elapsed_sec"])
        if not bytes_per_second or bytes_per_second <= 0:
            raise ValueError(f"{metrics_path} does not contain a positive byte throughput")
        estimated_seconds = target_bytes / bytes_per_second
        extrapolations[name] = {
            "benchmark_name": metrics_path.name,
            "bytes_per_second": bytes_per_second,
            "target_gb": args.target_gb,
            "estimated_seconds": estimated_seconds,
            "estimated_hours": estimated_seconds / 3600,
            "estimated_days": estimated_seconds / 86400,
        }

    result = {
        "input_name": args.input.name,
        "sample_strategy": "approximately evenly spaced documents aligned to delimiter boundaries",
        "document_count": len(documents),
        "delimiter": args.delimiter,
        "tokenizers": tokenizer_results,
        "throughput_extrapolations": extrapolations,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
