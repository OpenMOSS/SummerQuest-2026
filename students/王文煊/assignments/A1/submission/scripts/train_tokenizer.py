"""Train a byte-level BPE tokenizer and report basic statistics.

Example:
    python scripts/train_tokenizer.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --out artifacts/tinystories_tokenizer.pkl \
        --num-processes 64
"""

from __future__ import annotations

import argparse
import pickle
import time

from cs336_basics.tokenizer import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start = time.time()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    elapsed = time.time() - start

    with open(args.out, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": args.special_tokens}, f)

    longest = max(vocab.values(), key=len)
    print(f"[train_tokenizer] input={args.input}")
    print(f"[train_tokenizer] vocab_size={len(vocab)} merges={len(merges)}")
    print(f"[train_tokenizer] training_time_sec={elapsed:.2f}")
    print(f"[train_tokenizer] longest_token_len={len(longest)}")
    print(f"[train_tokenizer] longest_token={longest!r}")
    print(f"[train_tokenizer] saved -> {args.out}")


if __name__ == "__main__":
    main()
