from __future__ import annotations

import argparse
import array
import pickle
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream text into a flat uint16 token file")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.tokenizer.open("rb") as file:
        state = pickle.load(file)
    tokenizer = Tokenizer(**state)
    if max(tokenizer.vocab) > 65535:
        raise ValueError("uint16 output requires token IDs <= 65535")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    buffer = array.array("H")
    with args.input.open(encoding="utf-8") as source, args.output.open("wb") as destination:
        for token_id in tokenizer.encode_iterable(source):
            buffer.append(token_id)
            count += 1
            if len(buffer) >= 1_000_000:
                buffer.tofile(destination)
                buffer = array.array("H")
        buffer.tofile(destination)
    print(f"encoded {count} tokens to {args.output}")


if __name__ == "__main__":
    main()
