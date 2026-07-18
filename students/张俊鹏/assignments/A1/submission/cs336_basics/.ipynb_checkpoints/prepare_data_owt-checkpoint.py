from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import urllib.request
from multiprocessing import Pool, cpu_count

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


OWT_TRAIN_URL = (
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/"
    "owt_train.txt.gz"
)
OWT_VALID_URL = (
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/"
    "owt_valid.txt.gz"
)
END_OF_TEXT = "<|endoftext|>"

_worker_tokenizer = None
_worker_eos_id = None


def download(url: str, destination: str) -> None:
    if os.path.exists(destination):
        print(f"  {destination} already exists, skipping download")
        return

    temporary = destination + ".part"
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)
    print(f"  saved to {destination}")


def download_and_extract(url: str, text_path: str) -> None:
    if os.path.exists(text_path):
        print(f"  {text_path} already exists, skipping")
        return

    gzip_path = text_path + ".gz"
    download(url, gzip_path)

    temporary = text_path + ".part"
    print(f"  extracting {gzip_path}")
    with gzip.open(gzip_path, "rb") as source, open(temporary, "wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    os.replace(temporary, text_path)
    print(f"  extracted to {text_path}")


def iter_documents(text_path: str, block_size: int = 8 * 1024 * 1024):
    """Yield OWT documents separated by <|endoftext|> without loading all data."""
    remainder = ""
    with open(text_path, "r", encoding="utf-8") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break

            pieces = (remainder + block).split(END_OF_TEXT)
            remainder = pieces.pop()
            for document in pieces:
                if document:
                    yield document

    if remainder:
        yield remainder


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _initialize_worker(vocab, merges, special_tokens) -> None:
    global _worker_tokenizer, _worker_eos_id
    _worker_tokenizer = Tokenizer(vocab, merges, special_tokens)
    _worker_eos_id = _worker_tokenizer.inverse_vocab[b"<|endoftext|>"]


def _encode_documents(documents):
    token_ids = []
    for document in documents:
        token_ids.extend(_worker_tokenizer.encode(document))
        token_ids.append(_worker_eos_id)
    return np.asarray(token_ids, dtype=np.int32), len(documents)


def tokenize_and_save(
    tokenizer: Tokenizer,
    text_path: str,
    output_path: str,
    num_workers: int | None = None,
    documents_per_batch: int = 256,
) -> None:
    """Tokenize OWT in parallel and create a memory-mappable .npy file."""
    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    raw_path = output_path + ".tmp.bin"
    if os.path.exists(raw_path):
        os.remove(raw_path)

    print(f"  tokenizing {text_path} -> {output_path}")
    total_tokens = 0
    total_documents = 0
    document_batches = batched(iter_documents(text_path), documents_per_batch)

    with Pool(
        processes=num_workers,
        initializer=_initialize_worker,
        initargs=(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
    ) as pool, open(raw_path, "wb") as raw_file:
        for token_ids, document_count in pool.imap(
            _encode_documents, document_batches, chunksize=1
        ):
            token_ids.tofile(raw_file)
            total_tokens += token_ids.size
            total_documents += document_count

            if total_documents % 10_000 < document_count:
                print(
                    f"  processed {total_documents:,} documents, "
                    f"{total_tokens:,} tokens"
                )

    if total_tokens == 0:
        np.save(output_path, np.empty(0, dtype=np.int32))
        os.remove(raw_path)
        return

    print(f"  writing {total_tokens:,} tokens to NumPy file")
    raw = np.memmap(raw_path, mode="r", dtype=np.int32, shape=(total_tokens,))
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.int32, shape=(total_tokens,)
    )

    copy_size = 10_000_000
    for start in range(0, total_tokens, copy_size):
        end = min(start + copy_size, total_tokens)
        output[start:end] = raw[start:end]

    output.flush()
    del output, raw
    os.remove(raw_path)
    print(f"  saved {output_path}")


def save_bpe(vocab, merges, vocab_path: str, merges_path: str) -> None:
    serializable_vocab = {
        str(index): token.decode("iso-8859-1") for index, token in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as file:
        json.dump(serializable_vocab, file, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as file:
        for first, second in merges:
            file.write(
                f"{first.decode('iso-8859-1')} "
                f"{second.decode('iso-8859-1')}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenWebText data")
    parser.add_argument("--data_dir", default="data/owt")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--special_tokens", nargs="*", default=[END_OF_TEXT])
    parser.add_argument("--train_txt", default=None)
    parser.add_argument("--val_txt", default=None)
    parser.add_argument("--vocab_path", default=None)
    parser.add_argument("--merges_path", default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--documents_per_batch", type=int, default=256)
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    if END_OF_TEXT not in args.special_tokens:
        raise ValueError(f"special_tokens must include {END_OF_TEXT!r}")
    if bool(args.vocab_path) != bool(args.merges_path):
        raise ValueError("vocab_path and merges_path must be provided together")
    if args.train_txt and not args.val_txt:
        raise ValueError("val_txt is required when train_txt is provided")

    os.makedirs(args.data_dir, exist_ok=True)

    print("=== Step 1: Raw text ===")
    if args.train_txt:
        train_txt = args.train_txt
        val_txt = args.val_txt
    else:
        train_txt = os.path.join(args.data_dir, "owt_train.txt")
        val_txt = os.path.join(args.data_dir, "owt_valid.txt")
        download_and_extract(OWT_TRAIN_URL, train_txt)
        download_and_extract(OWT_VALID_URL, val_txt)

    vocab_path = args.vocab_path or os.path.join(args.data_dir, "vocab.json")
    merges_path = args.merges_path or os.path.join(args.data_dir, "merges.txt")

    print("=== Step 2: BPE tokenizer ===")
    should_train = not args.vocab_path and (
        args.force_retrain
        or not (os.path.exists(vocab_path) and os.path.exists(merges_path))
    )
    if should_train:
        print(f"  training BPE with vocab size {args.vocab_size:,}")
        vocab, merges = train_bpe(
            train_txt, args.vocab_size, args.special_tokens
        )
        save_bpe(vocab, merges, vocab_path, merges_path)
    else:
        print(f"  loading {vocab_path} and {merges_path}")

    tokenizer = Tokenizer.from_files(
        vocab_path, merges_path, args.special_tokens
    )
    print(f"  tokenizer vocabulary size: {len(tokenizer.vocab):,}")

    print("=== Step 3: Tokenization ===")
    train_output = os.path.join(args.data_dir, "owt_train.npy")
    val_output = os.path.join(args.data_dir, "owt_valid.npy")
    tokenize_and_save(
        tokenizer,
        train_txt,
        train_output,
        args.num_workers,
        args.documents_per_batch,
    )
    tokenize_and_save(
        tokenizer,
        val_txt,
        val_output,
        args.num_workers,
        args.documents_per_batch,
    )

    print("=== Done ===")
    print(f"Training data:   {train_output}")
    print(f"Validation data: {val_output}")


if __name__ == "__main__":
    main()
