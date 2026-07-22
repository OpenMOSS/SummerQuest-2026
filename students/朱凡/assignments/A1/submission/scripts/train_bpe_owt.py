"""Train BPE tokenizer on OpenWebText (vocab_size=32000) for problem train_bpe_expts_owt.

Reports time, longest token.
Serializes vocab/merges to data/ for later use.

Usage:
    uv run python scripts/train_bpe_owt.py
"""
from __future__ import annotations

import argparse
import base64
import json
import resource
import sys
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe

NUM_WORKERS = 32


def peak_rss_bytes() -> tuple[int, int]:
    """Return (main_peak_bytes, max_child_peak_bytes).

    ru_maxrss is the peak RSS over the process's lifetime (not instantaneous).
    macOS reports bytes; Linux reports KB.
    """
    main = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform != "darwin":
        main *= 1024
        children *= 1024
    return main, children

INPUT = Path("data/owt_train.txt")
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_OUT = Path("data/owt_vocab.json")
MERGES_OUT = Path("data/owt_merges.txt")


def save_tokenizer(vocab, merges, vocab_path: Path, merges_path: Path) -> None:
    with open(vocab_path, "w") as f:
        json.dump(
            {str(k): base64.b64encode(v).decode() for k, v in vocab.items()},
            f,
        )
    with open(merges_path, "w") as f:
        for b1, b2 in merges:
            f.write(f"{base64.b64encode(b1).decode()} {base64.b64encode(b2).decode()}\n")


def main(input_path: Path, vocab_size: int, vocab_out: Path, merges_out: Path, num_workers: int = NUM_WORKERS) -> None:
    t0 = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, SPECIAL_TOKENS, num_workers=num_workers)
    elapsed = time.time() - t0

    main_peak, child_peak = peak_rss_bytes()
    total_peak = main_peak + num_workers * child_peak

    save_tokenizer(vocab, merges, vocab_out, merges_out)

    longest = max(vocab.values(), key=len)

    print("=== train_bpe_owt ===")
    print(f"Input: {input_path}")
    print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print(f"Num workers: {num_workers}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Peak RSS (main process): {main_peak / 1024**3:.2f} GB")
    print(f"Peak RSS (max per worker): {child_peak / 1024**3:.2f} GB")
    print(f"Peak RSS (est. total = main + {num_workers} workers): {total_peak / 1024**3:.2f} GB")
    print(f"Longest token: {longest!r} ({len(longest)} bytes)")
    print(f"Saved vocab to {vocab_out}")
    print(f"Saved merges to {merges_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT, help="Input text file path")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE, help="Target vocab size")
    parser.add_argument("--vocab-out", type=Path, default=VOCAB_OUT, help="Output vocab.json path")
    parser.add_argument("--merges-out", type=Path, default=MERGES_OUT, help="Output merges.txt path")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of worker processes for pretoken counting")
    args = parser.parse_args()
    main(args.input, args.vocab_size, args.vocab_out, args.merges_out, args.num_workers)
