from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.bpe import Tokenizer


def load_hex_vocab(path: Path) -> dict[int, bytes]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in raw.items()}


def load_hex_merges(path: Path) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split()
        merges.append((bytes.fromhex(left), bytes.fromhex(right)))
    return merges


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text into a NumPy token-id array.")
    parser.add_argument("--input", required=True, help="Text file to encode.")
    parser.add_argument("--vocab", required=True, help="Hex JSON vocab produced by train_tokenizer.py.")
    parser.add_argument("--merges", required=True, help="Hex merges produced by train_tokenizer.py.")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32", "int64"])
    args = parser.parse_args()

    tokenizer = Tokenizer(load_hex_vocab(Path(args.vocab)), load_hex_merges(Path(args.merges)), args.special_token)
    with open(args.input, encoding="utf-8") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.dtype(args.dtype))
    np.save(args.output, token_ids)


if __name__ == "__main__":
    main()
