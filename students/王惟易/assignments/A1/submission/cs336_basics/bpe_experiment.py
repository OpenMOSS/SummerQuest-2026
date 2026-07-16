import argparse
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe
from cs336_basics.bpe_io import save_bpe

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and inspect a byte-level BPE tokenizer"
    )

    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)

    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True
    )

    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    special_tokens: list[str] = args.special_token

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )

    elapsed_seconds = time.perf_counter() - start_time

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_memory_mib = usage.ru_maxrss / 1024

    save_bpe(
        output_path=args.output_path,
        vocab=vocab,
        merges=merges,
    )

    longest_overall = max(
        vocab.values(),
        key=len,
    )

    special_token_bytes = {
        token.encode()
        for token in special_tokens
    }

    learned_tokens = (
        token for token in vocab.values() if token not in special_token_bytes
    )

    longest_learned = max(learned_tokens, key=len)

    print(f"input_path: {args.input_path}")
    print(f"requested vocab size: {args.vocab_size}")
    print(f"final vocab size: {len(vocab)}")
    print(f"number of merges: {len(merges)}")
    print(f"elapsed seconds: {elapsed_seconds:.3f}")
    print(f"peak RSS MiB: {peak_memory_mib:.2f}")

    print(
        f"longest overall token: {longest_overall!r} "
        f"({len(longest_overall)} bytes)"
    )

    print(
        f"longest learned token: {longest_learned!r} "
        f"({len(longest_learned)} bytes)"
    )

    print(f"serialized output: {args.output_path}")


if __name__ == "__main__":
    main()