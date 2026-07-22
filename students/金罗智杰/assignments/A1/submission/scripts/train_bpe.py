from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

from cs336_basics.artifacts import save_tokenizer
from cs336_basics.bpe import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"training tokenizer: input={args.input} vocab_size={args.vocab_size}",
        flush=True,
    )
    started = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    save_tokenizer(args.output, vocab, merges, args.special_token)
    elapsed = time.perf_counter() - started
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"saved tokenizer: {args.output}")
    print(
        f"vocab_size={len(vocab)} merges={len(merges)} "
        f"elapsed_seconds={elapsed:.3f} peak_rss_mib={peak_rss_mib:.1f}"
    )


if __name__ == "__main__":
    main()
