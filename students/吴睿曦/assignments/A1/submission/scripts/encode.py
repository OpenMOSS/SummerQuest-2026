import argparse
from itertools import chain
import multiprocessing as mp
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


_worker_tokenizer = None


def initialize_worker(vocab, merges, special_tokens):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(vocab, merges, special_tokens)


def encode_text(text):
    return _worker_tokenizer.encode(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode UTF-8 text as a NumPy token array")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--workers", type=int, default=1, help="Parallel line encoders (default: 1)")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    dtype = np.uint16 if len(tokenizer.vocab) <= 65536 else np.uint32
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as source:
        if args.workers == 1:
            ids = np.fromiter(tokenizer.encode_iterable(source), dtype=dtype)
        else:
            if args.workers < 1:
                raise ValueError("workers must be positive")
            with mp.Pool(
                args.workers,
                initializer=initialize_worker,
                initargs=(args.vocab, args.merges, args.special_token),
            ) as pool:
                encoded_lines = pool.imap(encode_text, source, chunksize=64)
                ids = np.fromiter(chain.from_iterable(encoded_lines), dtype=dtype)
        np.save(args.output, ids)


if __name__ == "__main__":
    main()
