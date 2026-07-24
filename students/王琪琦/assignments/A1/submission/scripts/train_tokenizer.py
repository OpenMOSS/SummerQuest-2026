from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tokenizer_io import save_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    save_tokenizer(Tokenizer(vocab, merges, args.special_token), args.output_dir)
    elapsed = time.perf_counter() - start
    summary = {
        "input_bytes": Path(args.input).stat().st_size,
        "vocab_size": len(vocab),
        "merges": len(merges),
        "training_time_sec": elapsed,
    }
    output_dir = Path(args.output_dir)
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"trained vocab_size={len(vocab)} merges={len(merges)} seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
