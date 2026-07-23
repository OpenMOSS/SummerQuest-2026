import argparse
import json
from pathlib import Path
import resource
import time

from cs336_basics.tokenizer import Tokenizer, train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    elapsed = time.perf_counter() - start
    Tokenizer(vocab, merges, args.special_token).save(
        args.output_dir / "vocab.json", args.output_dir / "merges.json"
    )
    longest = max(vocab.values(), key=len)
    summary = {
        "input_bytes": args.input.stat().st_size,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "wall_clock_sec": elapsed,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "longest_token_bytes": list(longest),
        "longest_token_length": len(longest),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
