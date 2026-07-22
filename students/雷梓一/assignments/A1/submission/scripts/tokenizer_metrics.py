from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


SPECIAL_TOKEN = "<|endoftext|>"


def sample_documents(path: Path, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    remainder = ""
    with path.open(encoding="utf-8") as file:
        while chunk := file.read(4 * 1024 * 1024):
            pieces = (remainder + chunk).split(SPECIAL_TOKEN)
            remainder = pieces.pop()
            for document in pieces:
                document = document.strip()
                if not document:
                    continue
                seen += 1
                if len(reservoir) < count:
                    reservoir.append(document)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < count:
                        reservoir[replacement] = document
    if remainder.strip():
        seen += 1
        if len(reservoir) < count:
            reservoir.append(remainder.strip())
        else:
            replacement = rng.randrange(seen)
            if replacement < count:
                reservoir[replacement] = remainder.strip()
    if len(reservoir) != count:
        raise ValueError(f"requested {count} documents from {path}, found only {seen}")
    return reservoir


def compression_metrics(tokenizer: Tokenizer, documents: list[str]) -> dict[str, float | int]:
    byte_count = sum(len(document.encode("utf-8")) for document in documents)
    token_count = sum(len(tokenizer.encode(document)) for document in documents)
    return {
        "documents": len(documents),
        "bytes": byte_count,
        "tokens": token_count,
        "bytes_per_token": byte_count / token_count,
    }


def throughput_metrics(tokenizer: Tokenizer, documents: list[str]) -> dict[str, float | int]:
    sample = (SPECIAL_TOKEN.join(documents) + SPECIAL_TOKEN) * 32
    sample_bytes = len(sample.encode("utf-8"))
    start = time.perf_counter()
    token_count = len(tokenizer.encode(sample))
    elapsed = time.perf_counter() - start
    bytes_per_second = sample_bytes / elapsed
    pile_bytes = 825 * 1024**3
    return {
        "benchmark_bytes": sample_bytes,
        "benchmark_tokens": token_count,
        "wall_clock_sec": elapsed,
        "bytes_per_second": bytes_per_second,
        "estimated_pile_hours": pile_bytes / bytes_per_second / 3600,
    }


def longest_token_metrics(tokenizer: Tokenizer) -> dict[str, float | int | str]:
    longest = b""
    for token in tokenizer.vocab.values():
        if len(token) > len(longest):
            longest = token
    return {
        "bytes": len(longest),
        "hex": longest.hex(),
        "text": longest.decode("utf-8", errors="replace"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure A1 tokenizer compression and throughput.")
    parser.add_argument("--tinystories", required=True, type=Path)
    parser.add_argument("--owt", required=True, type=Path)
    parser.add_argument("--tinystories-vocab", required=True, type=Path)
    parser.add_argument("--tinystories-merges", required=True, type=Path)
    parser.add_argument("--owt-vocab", required=True, type=Path)
    parser.add_argument("--owt-merges", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tiny_documents = sample_documents(args.tinystories, args.num_documents, args.seed)
    owt_documents = sample_documents(args.owt, args.num_documents, args.seed)
    tiny_tokenizer = Tokenizer.from_files(
        args.tinystories_vocab, args.tinystories_merges, [SPECIAL_TOKEN]
    )
    owt_tokenizer = Tokenizer.from_files(args.owt_vocab, args.owt_merges, [SPECIAL_TOKEN])
    tokenizers = {"tinystories": tiny_tokenizer, "owt": owt_tokenizer}
    datasets = {"tinystories": tiny_documents, "owt": owt_documents}
    metrics: dict[str, object] = {
        "num_sampled_documents": args.num_documents,
        "seed": args.seed,
        "compression": {
            f"{tokenizer_name}_tokenizer_on_{dataset_name}": compression_metrics(tokenizer, documents)
            for tokenizer_name, tokenizer in tokenizers.items()
            for dataset_name, documents in datasets.items()
        },
        "throughput": {
            tokenizer_name: throughput_metrics(tokenizer, tiny_documents + owt_documents)
            for tokenizer_name, tokenizer in tokenizers.items()
        },
        "longest_tokens": {
            tokenizer_name: longest_token_metrics(tokenizer)
            for tokenizer_name, tokenizer in tokenizers.items()
        },
        "sample_documents": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
