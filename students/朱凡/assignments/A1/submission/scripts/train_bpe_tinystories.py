"""Train BPE tokenizer on TinyStories (vocab_size=10000) for problem train_bpe_tinystories.

Reports time, peak memory, longest token.
Serializes vocab/merges to data/ for later use.
Optionally profiles with cProfile when run with --profile.
Optionally profiles each worker with --profile-workers (merged via pstats.Stats).

Usage:
    uv run python scripts/train_bpe_tinystories.py
    uv run python scripts/train_bpe_tinystories.py --profile
    uv run python scripts/train_bpe_tinystories.py --profile-workers
"""
from __future__ import annotations

import argparse
import base64
import cProfile
import json
import os
import pstats
import resource
import shutil
import sys
import time
from pathlib import Path

from cs336_basics.bpe import get_last_worker_prof_files, train_bpe

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

INPUT = Path("data/TinyStoriesV2-GPT4-train.txt")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_OUT = Path("data/tinystories_vocab.json")
MERGES_OUT = Path("data/tinystories_merges.txt")


def save_tokenizer(vocab, merges, vocab_path: Path, merges_path: Path) -> None:
    with open(vocab_path, "w") as f:
        json.dump(
            {str(k): base64.b64encode(v).decode() for k, v in vocab.items()},
            f,
        )
    with open(merges_path, "w") as f:
        for b1, b2 in merges:
            f.write(f"{base64.b64encode(b1).decode()} {base64.b64encode(b2).decode()}\n")


def main(profile: bool = False, profile_workers: bool = False, num_workers: int = NUM_WORKERS) -> None:
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    t0 = time.time()
    vocab, merges = train_bpe(
        INPUT, VOCAB_SIZE, SPECIAL_TOKENS, num_workers=num_workers, profile_workers=profile_workers
    )
    elapsed = time.time() - t0

    if profile:
        profiler.disable()
        print("\n=== Main process cProfile (top 15 cumulative) ===")
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(15)

    main_peak, child_peak = peak_rss_bytes()
    total_peak = main_peak + num_workers * child_peak

    save_tokenizer(vocab, merges, VOCAB_OUT, MERGES_OUT)

    if profile_workers:
        prof_files = get_last_worker_prof_files()
        if prof_files:
            print(f"\n=== Worker cProfile (merged {len(prof_files)} workers, top 20 by tottime) ===")
            stats = pstats.Stats(*prof_files)
            stats.sort_stats("tottime").print_stats(20)
            tmp_dir = os.path.dirname(prof_files[0])
            shutil.rmtree(tmp_dir, ignore_errors=True)

    longest = max(vocab.values(), key=len)

    print("\n=== train_bpe_tinystories ===")
    print(f"Input: {INPUT}")
    print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print(f"Num workers: {num_workers}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Peak RSS (main process): {main_peak / 1024**3:.2f} GB")
    print(f"Peak RSS (max per worker): {child_peak / 1024**3:.2f} GB")
    print(f"Peak RSS (est. total = main + {num_workers} workers): {total_peak / 1024**3:.2f} GB")
    print(f"Longest token: {longest!r} ({len(longest)} bytes)")
    print(f"Saved vocab to {VOCAB_OUT}")
    print(f"Saved merges to {MERGES_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="Run cProfile on main process and print top 15 cumulative")
    parser.add_argument("--profile-workers", action="store_true", help="Profile each worker (cProfile) and merge stats via pstats.Stats")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of worker processes for pretoken counting")
    args = parser.parse_args()
    main(profile=args.profile, profile_workers=args.profile_workers, num_workers=args.num_workers)
