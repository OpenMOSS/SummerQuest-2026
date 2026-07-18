"""Prepare TinyStories dataset for training.

1. Download / locate raw text
2. Train BPE or load existing vocab/merges
3. Tokenize with multiprocessing + save as .npy memmap
"""
import argparse
import json
import os
import urllib.request
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)
TINYSTORIES_VAL_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  {dest} already exists, skipping.")
        return
    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved to {dest}")


def _encode_chunk(args):
    """Encode a chunk of lines. Runs in a worker process."""
    chunk_lines, vocab, merges, special_tokens = args
    tok = Tokenizer(vocab, merges, special_tokens)
    eos_id = tok.inverse_vocab.get(b"<|endoftext|>", tok.inverse_vocab.get(b"\n", 0))
    ids = []
    for line in chunk_lines:
        line = line.strip()
        if not line:
            ids.append(eos_id)
            continue
        ids.extend(tok.encode(line))
        ids.append(eos_id)
    return ids


def tokenize_and_save(tokenizer: Tokenizer, text_path: str, out_path: str, num_workers: int = None):
    """Tokenize a text file in parallel and save as int32 .npy."""
    print(f"  tokenizing {text_path} -> {out_path} ...")

    # Read all lines
    with open(text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  {len(lines):,} lines read")

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    # Split into chunks for workers
    chunk_size = max(1, len(lines) // num_workers)
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # Prepare args for each worker
    worker_args = [(chunk, tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)
                   for chunk in chunks]

    print(f"  encoding with {len(chunks)} workers ...")
    with Pool(processes=len(chunks)) as pool:
        results = pool.map(_encode_chunk, worker_args)

    # Flatten
    ids = []
    for r in results:
        ids.extend(r)

    print(f"  {len(ids):,} tokens, saving ...")
    arr = np.array(ids, dtype=np.int32)
    np.save(out_path, arr)
    print(f"  saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories data.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--special_tokens", type=str, nargs="*",
                        default=["<|endoftext|>"])
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--train_txt", type=str, default=None,
                        help="Path to existing training text (skip download)")
    parser.add_argument("--val_txt", type=str, default=None,
                        help="Path to existing validation text (skip download)")
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--merges_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # 1. Get raw text
    print("=== Step 1: Raw text ===")
    if args.train_txt:
        train_txt = args.train_txt
        val_txt = args.val_txt or args.train_txt
        print(f"  using existing: {train_txt}")
        print(f"  using existing: {val_txt}")
    else:
        train_txt = os.path.join(args.data_dir, "TinyStories-train.txt")
        val_txt = os.path.join(args.data_dir, "TinyStories-valid.txt")
        download(TINYSTORIES_URL, train_txt)
        download(TINYSTORIES_VAL_URL, val_txt)

    # 2. Load BPE tokenizer
    if args.vocab_path and args.merges_path:
        vocab_path = args.vocab_path
        merges_path = args.merges_path
    else:
        vocab_path = os.path.join(args.data_dir, "vocab.json")
        merges_path = os.path.join(args.data_dir, "merges.txt")

    if args.vocab_path and args.merges_path:
        print(f"=== Step 2: Use existing BPE ===")
        print(f"  vocab: {vocab_path}")
        print(f"  merges: {merges_path}")
    elif args.force_retrain or not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
        print(f"=== Step 2: Train BPE (vocab_size={args.vocab_size}) ===")
        vocab, merges = train_bpe(train_txt, args.vocab_size, args.special_tokens)
        serializable = {str(k): v.decode("iso-8859-1") for k, v in vocab.items()}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in merges:
                f.write(f"{a.decode('iso-8859-1')} {b.decode('iso-8859-1')}\n")
        print(f"  vocab saved: {vocab_path}")
        print(f"  merges saved: {merges_path}")
    else:
        print(f"=== Step 2: Load existing BPE ===")
        print(f"  using {vocab_path} and {merges_path}")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, args.special_tokens)
    print(f"  vocab size: {len(tokenizer.vocab)}")

    # 3. Tokenize (parallel)
    print("=== Step 3: Tokenize ===")
    train_npy = os.path.join(args.data_dir, "tinystories_train.npy")
    val_npy = os.path.join(args.data_dir, "tinystories_val.npy")

    tokenize_and_save(tokenizer, train_txt, train_npy, args.num_workers)
    tokenize_and_save(tokenizer, val_txt, val_npy, args.num_workers)

    print("=== Done ===")
    print(f"Training data:  {train_npy}")
    print(f"Validation data: {val_npy}")


if __name__ == "__main__":
    main()