from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tokenizer encoding on a fixed byte prefix")
    parser.add_argument("--name", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.tokenizer.open("rb") as file:
        state = pickle.load(file)
    tokenizer = Tokenizer(**state)
    with args.input.open("rb") as file:
        raw = file.read(args.sample_bytes)
    text = raw.decode("utf-8", errors="ignore")
    measured_bytes = len(text.encode("utf-8"))

    started = time.perf_counter()
    token_ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - started
    special_bytes = {token.encode("utf-8") for token in state["special_tokens"]}
    ordinary_tokens = [value for value in state["vocab"].values() if value not in special_bytes]
    longest = max(ordinary_tokens, key=len)
    result = {
        "name": args.name,
        "sample_bytes": measured_bytes,
        "encoded_tokens": len(token_ids),
        "elapsed_sec": elapsed,
        "bytes_per_token": measured_bytes / len(token_ids),
        "bytes_per_sec": measured_bytes / elapsed,
        "tokens_per_sec": len(token_ids) / elapsed,
        "vocab_size": len(state["vocab"]),
        "longest_token_bytes": len(longest),
        "longest_token_repr": repr(longest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
