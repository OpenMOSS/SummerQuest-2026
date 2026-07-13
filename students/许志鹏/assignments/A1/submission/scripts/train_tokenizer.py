from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.tokenizer import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True, help="Path to UTF-8 training text.")
    parser.add_argument("--output", required=True, help="Output tokenizer pickle path.")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--metadata", default=None, help="Optional JSON metadata output path.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for pre-tokenization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens, num_workers=args.num_workers)
    elapsed = time.time() - start
    payload = {
        "vocab": vocab,
        "merges": merges,
        "special_tokens": special_tokens,
    }
    with output.open("wb") as f:
        pickle.dump(payload, f)

    metadata = {
        "input": str(args.input),
        "output": str(output),
        "vocab_size": args.vocab_size,
        "special_tokens": special_tokens,
        "num_merges": len(merges),
        "num_workers": args.num_workers,
        "elapsed_sec": elapsed,
    }
    if args.metadata is not None:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
