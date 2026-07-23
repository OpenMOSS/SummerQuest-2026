from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import time
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
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer(vocab, merges, special_tokens)


def _encode_documents(documents):
    # Encoding one joined string avoids one Python/tokenizer call per document.
    # Since END_OF_TEXT is registered as a special token, this is equivalent to
    # encoding each document separately and appending the EOS token after it.
    text = END_OF_TEXT.join(documents) + END_OF_TEXT
    token_ids = _worker_tokenizer.encode(text)
    return np.asarray(token_ids, dtype=np.int32), len(documents)


def _npy_header(shape: tuple[int, ...]) -> bytes:
    """Build a standard NumPy v2 header for a C-contiguous int32 array."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.int32)),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return buffer.getvalue()


def tokenize_and_save(
    tokenizer: Tokenizer,
    text_path: str,
    output_path: str,
    num_workers: int | None = None,
    documents_per_batch: int = 1024,
    force: bool = False,
) -> None:
    """Tokenize OWT in parallel and create a memory-mappable .npy file."""
    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    if os.path.exists(output_path) and not force:
        print(f"  {output_path} already exists, skipping tokenization")
        return

    temporary_path = output_path + ".part"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    print(f"  tokenizing {text_path} -> {output_path}")
    started_at = time.perf_counter()
    total_tokens = 0
    total_documents = 0
    document_batches = batched(iter_documents(text_path), documents_per_batch)
    placeholder_header = _npy_header((0,))

    with Pool(
        processes=num_workers,
        initializer=_initialize_worker,
        initargs=(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
    ) as pool, open(temporary_path, "w+b") as output_file:
        # Reserve the header and stream token data immediately after it. Once
        # the final length is known, the header is rewritten in place.
        output_file.write(placeholder_header)
        for token_ids, document_count in pool.imap(
            _encode_documents, document_batches, chunksize=1
        ):
            token_ids.tofile(output_file)
            total_tokens += token_ids.size
            total_documents += document_count

            if total_documents % 10_000 < document_count:
                elapsed = time.perf_counter() - started_at
                print(
                    f"  processed {total_documents:,} documents, "
                    f"{total_tokens:,} tokens "
                    f"({total_tokens / max(elapsed, 1e-9):,.0f} tokens/s)"
                )

        final_header = _npy_header((total_tokens,))
        if len(final_header) != len(placeholder_header):
            raise RuntimeError("NumPy header size changed while finalizing output")
        output_file.seek(0)
        output_file.write(final_header)
        output_file.flush()

    os.replace(temporary_path, output_path)
    elapsed = time.perf_counter() - started_at
    print(
        f"  saved {output_path}: {total_tokens:,} tokens in {elapsed:.1f}s "
        f"({total_tokens / max(elapsed, 1e-9):,.0f} tokens/s)"
    )


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
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--documents_per_batch", type=int, default=1024)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--force_retokenize", action="store_true")
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
        args.force_retokenize,
    )
    tokenize_and_save(
        tokenizer,
        val_txt,
        val_output,
        args.num_workers,
        args.documents_per_batch,
        args.force_retokenize,
    )

    print("=== Done ===")
    print(f"Training data:   {train_output}")
    print(f"Validation data: {val_output}")


if __name__ == "__main__":
    main()