"""Prepare tokenized datasets for training."""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from cs336_basics.tokenizer import Tokenizer, run_train_bpe


def _encode_lines(args: Tuple[List[str], str, str, List[str]]) -> np.ndarray:
    """Worker: encode a chunk of text lines into token IDs."""
    lines, vocab_path, merges_path, special_tokens = args
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    token_ids: List[int] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        token_ids.extend(tokenizer.encode(line + "<|endoftext|>"))
    return np.array(token_ids, dtype=np.uint16)


def _line_chunks(path: str, chunk_size: int) -> Iterable[List[str]]:
    """Yield chunks of non-empty lines from a text file."""
    chunk: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def _tokenize_split(
    text_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: List[str],
    output_path: Path,
    num_workers: int,
    chunk_size: int,
) -> None:
    """Tokenize a text split using multiple workers and save as .npy."""
    parts: List[np.ndarray] = []
    total_tokens = 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        # Submit chunks as they are read to bound memory usage.
        futures = []
        for chunk in _line_chunks(text_path, chunk_size):
            future = executor.submit(
                _encode_lines,
                (chunk, str(vocab_path), str(merges_path), special_tokens),
            )
            futures.append(future)
            if len(futures) > num_workers * 4:
                # Collect completed results to keep memory bounded.
                done, futures = futures[:num_workers], futures[num_workers:]
                for future in done:
                    arr = future.result()
                    parts.append(arr)
                    total_tokens += len(arr)
        for future in futures:
            arr = future.result()
            parts.append(arr)
            total_tokens += len(arr)

    token_ids_arr = np.concatenate(parts) if parts else np.array([], dtype=np.uint16)
    np.save(output_path, token_ids_arr)
    print(f"Saved {total_tokens:,} tokens to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and encode datasets.")
    parser.add_argument("--train_text", type=str, required=True, help="Path to training text file.")
    parser.add_argument("--val_text", type=str, required=True, help="Path to validation text file.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Target vocabulary size.")
    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--chunk_size", type=int, default=10_000)
    parser.add_argument("--min_frequency", type=int, default=1, help="Minimum pre-token frequency to include in BPE training.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    print("Training BPE tokenizer...")
    vocab, merges = run_train_bpe(
        args.train_text,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        min_frequency=args.min_frequency,
    )
    tokenizer = Tokenizer(vocab, merges, special_tokens=args.special_tokens)
    tokenizer.save(str(vocab_path), str(merges_path))
    print(f"Saved tokenizer to {output_dir}")

    for split, text_path in [("train", args.train_text), ("val", args.val_text)]:
        print(f"Tokenizing {split} set with {args.num_workers} workers...")
        _tokenize_split(
            text_path,
            vocab_path,
            merges_path,
            args.special_tokens,
            output_dir / f"{split}.npy",
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
