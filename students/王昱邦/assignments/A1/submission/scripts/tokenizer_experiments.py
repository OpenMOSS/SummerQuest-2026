from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
END_OF_TEXT = "<|endoftext|>"

TINYSTORIES_VALID_PATH = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
OWT_VALID_PATH = PROJECT_ROOT / "data" / "owt_valid.txt"

TINYSTORIES_TOKENIZER_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tinystories_bpe"
    / "train"
    / "vocab_10000"
    / "tinystories_train_10000_v1"
)
OWT_TOKENIZER_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "owt_bpe"
    / "owt-train"
    / "vocab_32000"
    / "owt_train_32000_v1"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "tokenizer_experiments" / "sample_analysis.json"
)


def iter_documents(
    input_path: Path,
    read_size: int = 1024 * 1024,
) -> Iterator[str]:
    """Yield documents separated by END_OF_TEXT without loading the file at once."""
    remainder = ""

    with input_path.open("r", encoding="utf-8") as input_file:
        while chunk := input_file.read(read_size):
            segments = (remainder + chunk).split(END_OF_TEXT)
            remainder = segments.pop()

            for document in segments:
                if document:
                    yield document

    if remainder:
        yield remainder


def reservoir_sample(
    documents: Iterator[str],
    sample_size: int,
    seed: int,
) -> tuple[list[str], int]:
    """Uniformly sample documents from a stream using bounded memory."""
    rng = random.Random(seed)
    samples: list[str] = []
    document_count = 0

    for document_count, document in enumerate(documents, start=1):
        if document_count <= sample_size:
            samples.append(document)
            continue

        replacement_index = rng.randrange(document_count)
        if replacement_index < sample_size:
            samples[replacement_index] = document

    if document_count < sample_size:
        raise ValueError(
            f"Only found {document_count} documents; cannot sample {sample_size}."
        )

    return samples, document_count


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_files(
        str(tokenizer_dir / "vocab.pkl"),
        str(tokenizer_dir / "merges.pkl"),
        special_tokens=[END_OF_TEXT],
    )


def compression_metrics(tokenizer: Tokenizer, documents: list[str]) -> dict[str, float | int]:
    byte_count = sum(len(document.encode("utf-8")) for document in documents)
    token_count = sum(len(tokenizer.encode(document)) for document in documents)

    return {
        "documents": len(documents),
        "bytes": byte_count,
        "tokens": token_count,
        "bytes_per_token": byte_count / token_count,
    }


def throughput_metrics(
    tokenizer: Tokenizer,
    documents: list[str],
    target_bytes: int,
) -> dict[str, float | int]:
    benchmark_text = END_OF_TEXT.join(documents)
    bytes_per_round = len(benchmark_text.encode("utf-8"))
    rounds = max(3, math.ceil(target_bytes / bytes_per_round))

    tokenizer.encode(benchmark_text)

    start_time = time.perf_counter()
    token_count = 0
    for _ in range(rounds):
        token_count += len(tokenizer.encode(benchmark_text))
    elapsed_seconds = time.perf_counter() - start_time

    total_bytes = bytes_per_round * rounds
    bytes_per_second = total_bytes / elapsed_seconds
    pile_bytes = 825 * 10**9

    return {
        "benchmark_rounds": rounds,
        "bytes": total_bytes,
        "tokens": token_count,
        "elapsed_seconds": elapsed_seconds,
        "bytes_per_second": bytes_per_second,
        "pile_825gb_estimated_seconds": pile_bytes / bytes_per_second,
        "pile_825gb_estimated_hours": pile_bytes / bytes_per_second / 3600,
        "pile_825gb_estimated_days": pile_bytes / bytes_per_second / 86400,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tokenizer experiments from section 2.7.")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--throughput-bytes", type=int, default=5_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for required_path in (
        TINYSTORIES_VALID_PATH,
        OWT_VALID_PATH,
        TINYSTORIES_TOKENIZER_DIR / "vocab.pkl",
        TINYSTORIES_TOKENIZER_DIR / "merges.pkl",
        OWT_TOKENIZER_DIR / "vocab.pkl",
        OWT_TOKENIZER_DIR / "merges.pkl",
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    print("Sampling TinyStories validation documents...")
    tiny_samples, tiny_document_count = reservoir_sample(
        iter_documents(TINYSTORIES_VALID_PATH), args.sample_size, args.seed
    )
    print("Sampling OpenWebText validation documents...")
    owt_samples, owt_document_count = reservoir_sample(
        iter_documents(OWT_VALID_PATH), args.sample_size, args.seed
    )

    print("Loading trained tokenizers...")
    tiny_tokenizer = load_tokenizer(TINYSTORIES_TOKENIZER_DIR)
    owt_tokenizer = load_tokenizer(OWT_TOKENIZER_DIR)

    results = {
        "configuration": {
            "sample_size": args.sample_size,
            "seed": args.seed,
            "throughput_target_bytes": args.throughput_bytes,
            "document_delimiter": END_OF_TEXT,
            "sampling_method": "reservoir sampling over validation documents",
        },
        "corpora": {
            "tinystories": {
                "path": str(TINYSTORIES_VALID_PATH),
                "document_count": tiny_document_count,
            },
            "openwebtext": {
                "path": str(OWT_VALID_PATH),
                "document_count": owt_document_count,
            },
        },
        "compression": {
            "tinystories_with_tinystories_tokenizer": compression_metrics(
                tiny_tokenizer, tiny_samples
            ),
            "openwebtext_with_openwebtext_tokenizer": compression_metrics(
                owt_tokenizer, owt_samples
            ),
            "openwebtext_with_tinystories_tokenizer": compression_metrics(
                tiny_tokenizer, owt_samples
            ),
        },
        "throughput": {
            "tinystories_tokenizer_on_tinystories": throughput_metrics(
                tiny_tokenizer, tiny_samples, args.throughput_bytes
            ),
            "openwebtext_tokenizer_on_openwebtext": throughput_metrics(
                owt_tokenizer, owt_samples, args.throughput_bytes
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {args.output}")
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
