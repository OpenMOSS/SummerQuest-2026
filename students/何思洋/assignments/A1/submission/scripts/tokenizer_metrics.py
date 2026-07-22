from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cs336_basics.tokenizer import load_tokenizer_spec, make_tokenizer


def tokenizer_metrics(
    tokenizer_path: Path,
    input_path: Path,
    max_bytes: int | None = None,
) -> dict[str, float | int | str]:
    vocab, merges, special_tokens = load_tokenizer_spec(tokenizer_path)
    tokenizer = make_tokenizer(vocab, merges, special_tokens)

    byte_count = 0
    token_count = 0
    start = time.perf_counter()
    with open(input_path, encoding="utf-8") as f:
        for chunk in f:
            if max_bytes is not None and byte_count >= max_bytes:
                break
            if max_bytes is not None:
                encoded_chunk = chunk.encode("utf-8")
                remaining = max_bytes - byte_count
                if len(encoded_chunk) > remaining:
                    chunk = encoded_chunk[:remaining].decode("utf-8", errors="ignore")
            byte_count += len(chunk.encode("utf-8"))
            token_count += sum(1 for _ in tokenizer.encode(chunk))
    elapsed = time.perf_counter() - start

    longest = max(vocab.values(), key=len)
    return {
        "tokenizer_path": str(tokenizer_path),
        "metric_text_path": str(input_path),
        "metric_max_bytes": max_bytes or 0,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "bytes": byte_count,
        "tokens": token_count,
        "compression_ratio_bytes_per_token": byte_count / token_count if token_count else 0,
        "throughput_tokens_per_sec": token_count / elapsed if elapsed else 0,
        "throughput_bytes_per_sec": byte_count / elapsed if elapsed else 0,
        "wall_clock_sec": elapsed,
        "longest_token_bytes": len(longest),
        "longest_token_utf8": longest.decode("utf-8", errors="replace"),
        "longest_token_hex": longest.hex(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tokenizer compression and throughput metrics.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    args = parser.parse_args()

    summary = tokenizer_metrics(args.tokenizer, args.input, args.max_bytes)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
