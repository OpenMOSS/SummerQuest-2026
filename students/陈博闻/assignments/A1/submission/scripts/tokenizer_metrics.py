from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.bpe import Tokenizer
from scripts.encode_dataset import load_hex_merges, load_hex_vocab


def file_byte_count(path: Path) -> int:
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tokenizer compression and encoding throughput.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    input_path = Path(args.input)
    tokenizer = Tokenizer(load_hex_vocab(Path(args.vocab)), load_hex_merges(Path(args.merges)), args.special_token)
    start = time.time()
    token_count = 0
    with input_path.open(encoding="utf-8") as f:
        for _ in tokenizer.encode_iterable(f):
            token_count += 1
    elapsed = time.time() - start
    byte_count = file_byte_count(input_path)
    longest_token_id, longest_token = max(tokenizer.vocab.items(), key=lambda item: len(item[1]))
    metrics = {
        "input": str(input_path),
        "byte_count": byte_count,
        "token_count": token_count,
        "compression_ratio_bytes_per_token": byte_count / token_count if token_count else None,
        "throughput_bytes_per_sec": byte_count / elapsed if elapsed > 0 else None,
        "throughput_tokens_per_sec": token_count / elapsed if elapsed > 0 else None,
        "elapsed_sec": elapsed,
        "longest_token_id": longest_token_id,
        "longest_token_num_bytes": len(longest_token),
        "longest_token_utf8_preview": longest_token.decode("utf-8", errors="replace")[:120],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
