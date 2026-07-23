from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cs336_basics.tokenizer import Tokenizer, train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--valid-text", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--special-token", default="<|endoftext|>")
    return parser.parse_args()


def encode_file(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> dict[str, float | int]:
    start = time.time()
    ids: list[int] = []
    byte_count = 0
    with open(input_path, encoding="utf-8") as file:
        for line in file:
            byte_count += len(line.encode("utf-8"))
            ids.extend(tokenizer.encode(line))
    dtype = np.uint16 if max(ids, default=0) < 2**16 else np.uint32
    np.save(output_path, np.asarray(ids, dtype=dtype))
    elapsed = time.time() - start
    return {
        "tokens": len(ids),
        "bytes": byte_count,
        "bytes_per_token": byte_count / max(len(ids), 1),
        "tokens_per_sec": len(ids) / max(elapsed, 1e-9),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = [args.special_token]
    start = time.time()
    tokenizer_path = out_dir / "tokenizer.pkl"
    if tokenizer_path.exists():
        with open(tokenizer_path, "rb") as file:
            tokenizer_payload = pickle.load(file)
        vocab = tokenizer_payload["vocab"]
        merges = tokenizer_payload["merges"]
        special_tokens = tokenizer_payload["special_tokens"]
    else:
        vocab, merges = train_bpe(args.train_text, args.vocab_size, special_tokens)
        with open(tokenizer_path, "wb") as file:
            pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, file)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    train_stats = encode_file(tokenizer, Path(args.train_text), out_dir / "train.npy")
    valid_stats = encode_file(tokenizer, Path(args.valid_text), out_dir / "valid.npy")
    longest_token = max(vocab.values(), key=len)
    summary = {
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "train_text": str(Path(args.train_text).name),
        "valid_text": str(Path(args.valid_text).name),
        "train": train_stats,
        "valid": valid_stats,
        "longest_token_bytes": len(longest_token),
        "longest_token_preview": longest_token.decode("utf-8", errors="replace"),
        "wall_clock_sec": time.time() - start,
    }
    with open(out_dir / "tokenizer_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
