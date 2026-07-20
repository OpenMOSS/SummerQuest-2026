from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic token-id .npy file for smoke tests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32", "int64"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, args.vocab_size, size=args.num_tokens, dtype=np.dtype(args.dtype))
    np.save(args.output, data)


if __name__ == "__main__":
    main()
