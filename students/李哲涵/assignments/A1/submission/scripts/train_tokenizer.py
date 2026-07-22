from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from _project_api import get_train_bpe
from _tokenizer_io import save_merges, save_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--vocab-out", type=Path, required=True)
    parser.add_argument("--merges-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    return parser.parse_args()


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS 返回 bytes；Linux 返回 KiB。
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return float(value) / divisor


def main() -> None:
    args = parse_args()
    if not args.input_path.is_file():
        raise FileNotFoundError(args.input_path)
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size 必须为正整数")

    start_time = time.perf_counter()
    start_rss_mb = peak_rss_mb()

    train_bpe = get_train_bpe()
    vocab, merges = train_bpe(
        str(args.input_path),
        args.vocab_size,
        list(args.special_token),
    )

    elapsed_seconds = time.perf_counter() - start_time
    end_peak_rss_mb = peak_rss_mb()

    if not vocab:
        raise RuntimeError("train_bpe 返回了空 vocab")

    longest_id, longest_bytes = max(vocab.items(), key=lambda item: len(item[1]))
    save_vocab(vocab, args.vocab_out)
    save_merges(merges, args.merges_out)

    metrics = {
        "input_path": str(args.input_path),
        "elapsed_seconds": elapsed_seconds,
        "start_peak_rss_mb": start_rss_mb,
        "peak_rss_mb": end_peak_rss_mb,
        "peak_rss_increase_mb": max(0.0, end_peak_rss_mb - start_rss_mb),
        "requested_vocab_size": args.vocab_size,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_id": int(longest_id),
        "longest_token": longest_bytes.decode("utf-8", errors="replace"),
        "longest_token_hex": longest_bytes.hex(),
        "longest_token_len": len(longest_bytes),
        "special_tokens": list(args.special_token),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"vocab_size={len(vocab)} num_merges={len(merges)}")
    print(
        f"longest_token={metrics['longest_token']!r} "
        f"len={metrics['longest_token_len']} hex={metrics['longest_token_hex']}"
    )
    print(f"elapsed={elapsed_seconds:.3f}s peak_rss={end_peak_rss_mb:.2f} MiB")
    print(f"saved vocab:   {args.vocab_out}")
    print(f"saved merges:  {args.merges_out}")
    print(f"saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
