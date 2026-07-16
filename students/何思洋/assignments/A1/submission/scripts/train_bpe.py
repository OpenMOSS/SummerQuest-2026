from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cs336_basics.tokenizer import make_tokenizer, save_tokenizer, train_bpe


def tokenizer_metrics(
    tokenizer_path: Path, input_path: Path, max_bytes: int | None = None
) -> dict[str, float | int | str]:
    from cs336_basics.tokenizer import load_tokenizer_spec

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
        "metric_text_path": str(input_path),
        "metric_max_bytes": max_bytes or 0,
        "bytes": byte_count,
        "tokens": token_count,
        "compression_ratio_bytes_per_token": byte_count / token_count if token_count else 0,
        "throughput_tokens_per_sec": token_count / elapsed if elapsed else 0,
        "throughput_bytes_per_sec": byte_count / elapsed if elapsed else 0,
        "longest_token_bytes": len(longest),
        "longest_token_utf8": longest.decode("utf-8", errors="replace"),
        "longest_token_hex": longest.hex(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Tokenizer JSON output path.")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--metrics-text", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--metrics-max-bytes", type=int, default=None)
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()

    def progress(stage: str, payload: dict[str, int]) -> None:
        record = {"stage": stage, "wall_clock_sec": time.perf_counter() - start, **payload}
        print(json.dumps(record), file=sys.stderr, flush=True)

    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token, progress_callback=progress)
    train_elapsed = time.perf_counter() - start
    save_tokenizer(args.output, vocab, merges, args.special_token)

    summary: dict[str, float | int | str | list[str]] = {
        "input_path": str(args.input),
        "tokenizer_path": str(args.output),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_token,
        "train_wall_clock_sec": train_elapsed,
    }

    if not args.skip_metrics:
        metrics_text = args.metrics_text or args.input
        summary.update(tokenizer_metrics(args.output, metrics_text, args.metrics_max_bytes))

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
