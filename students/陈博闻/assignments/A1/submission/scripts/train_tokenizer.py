from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.bpe import train_bpe


def write_vocab(path: Path, vocab: dict[int, bytes]) -> None:
    serializable = {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def write_merges(path: Path, merges: list[tuple[bytes, bytes]]) -> None:
    lines = [f"{left.hex()} {right.hex()}" for left, right in merges]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True, help="Training text file.")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--vocab-out", required=True)
    parser.add_argument("--merges-out", required=True)
    args = parser.parse_args()

    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    write_vocab(Path(args.vocab_out), vocab)
    write_merges(Path(args.merges_out), merges)


if __name__ == "__main__":
    main()
