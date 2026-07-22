"""Train a BPE tokenizer and encode TinyStories into memory-mappable token files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer, _find_chunk_boundaries, train_bpe


DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"


def save_tokenizer(tokenizer: Tokenizer, path: str | os.PathLike[str]) -> None:
    """Save byte-valued tokenizer state in a portable, non-pickle JSON format."""
    payload = {
        "format_version": 1,
        "special_tokens": tokenizer.special_tokens,
        "vocab": [[token_id, token.hex()] for token_id, token in sorted(tokenizer.vocab.items())],
        "merges": [[left.hex(), right.hex()] for left, right in tokenizer.merges],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def load_tokenizer(path: str | os.PathLike[str]) -> Tokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported tokenizer format in {path}")
    vocab = {int(token_id): bytes.fromhex(token) for token_id, token in payload["vocab"]}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in payload["merges"]]
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=payload.get("special_tokens", []))


def _token_dtype(tokenizer: Tokenizer) -> np.dtype:
    maximum_id = max(tokenizer.vocab)
    if maximum_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum_id <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError(f"Token id {maximum_id} does not fit in uint32")


def _encode_file_chunk(
    args: tuple[int, Tokenizer, str, int, int, str, str],
) -> tuple[int, int, int]:
    chunk_index, tokenizer, input_path, start, end, output_path, dtype_name = args
    dtype = np.dtype(dtype_name)
    token_count = 0
    with open(input_path, "rb") as source, open(output_path, "wb") as destination:
        source.seek(start)
        while source.tell() < end:
            remaining = end - source.tell()
            line = source.readline(remaining)
            if not line:
                break
            token_ids = tokenizer.encode(line.decode("utf-8"))
            if token_ids:
                np.asarray(token_ids, dtype=dtype).tofile(destination)
                token_count += len(token_ids)
    return chunk_index, token_count, end - start


def _encode_text_file_parallel(
    tokenizer: Tokenizer,
    input_path: Path,
    temporary_path: Path,
    dtype: np.dtype,
) -> int:
    worker_count = min(cpu_count(), 16)
    split_token = max(tokenizer.special_tokens, key=len).encode("utf-8")
    boundaries = _find_chunk_boundaries(input_path, worker_count, split_token)
    progress = tqdm(total=input_path.stat().st_size, unit="B", unit_scale=True, desc=f"Encoding {input_path.name}")
    token_counts = [0] * (len(boundaries) - 1)
    with tempfile.TemporaryDirectory(prefix=f".{temporary_path.name}-", dir=temporary_path.parent) as parts_dir:
        part_paths = [Path(parts_dir) / f"part-{index:04d}.bin" for index in range(len(token_counts))]
        tasks = [
            (
                index,
                tokenizer,
                str(input_path),
                start,
                end,
                str(part_paths[index]),
                dtype.name,
            )
            for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]
        try:
            with Pool(processes=len(tasks)) as pool:
                for index, token_count, bytes_processed in pool.imap_unordered(_encode_file_chunk, tasks):
                    token_counts[index] = token_count
                    progress.update(bytes_processed)
        finally:
            progress.close()

        with temporary_path.open("wb") as destination:
            for part_path in part_paths:
                with part_path.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    return sum(token_counts)


def encode_text_file(
    tokenizer: Tokenizer,
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    force: bool = False,
) -> int:
    """Stream a UTF-8 text file through the tokenizer and write raw token IDs."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dtype = _token_dtype(tokenizer)
    if input_path.stat().st_size >= 64 * 1024 * 1024 and tokenizer.special_tokens:
        token_count = _encode_text_file_parallel(tokenizer, input_path, temporary_path, dtype)
    else:
        token_count = 0
        with input_path.open("r", encoding="utf-8") as source, temporary_path.open("wb") as destination:
            progress = tqdm(
                total=input_path.stat().st_size,
                unit="B",
                unit_scale=True,
                desc=f"Encoding {input_path.name}",
            )
            try:
                for line in source:
                    token_ids = tokenizer.encode(line)
                    if token_ids:
                        np.asarray(token_ids, dtype=dtype).tofile(destination)
                        token_count += len(token_ids)
                    progress.update(len(line.encode("utf-8")))
            finally:
                progress.close()

    temporary_path.replace(output_path)
    return token_count


def prepare_dataset(
    train_text: str | os.PathLike[str],
    valid_text: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    vocab_size: int = 10_000,
    special_tokens: Iterable[str] = (DEFAULT_SPECIAL_TOKEN,),
    force: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    train_tokens_path = output_dir / "train.bin"
    valid_tokens_path = output_dir / "valid.bin"
    metadata_path = output_dir / "metadata.json"

    protected_outputs = (tokenizer_path, train_tokens_path, valid_tokens_path, metadata_path)
    existing_outputs = [str(path) for path in protected_outputs if path.exists()]
    if existing_outputs and not force:
        raise FileExistsError(
            "Prepared outputs already exist; pass --force to replace them: " + ", ".join(existing_outputs)
        )

    started_at = time.perf_counter()
    special_tokens = list(special_tokens)
    vocab, merges = train_bpe(train_text, vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    save_tokenizer(tokenizer, tokenizer_path)

    train_token_count = encode_text_file(tokenizer, train_text, train_tokens_path, force=True)
    valid_token_count = encode_text_file(tokenizer, valid_text, valid_tokens_path, force=True)
    dtype = _token_dtype(tokenizer)
    metadata: dict[str, object] = {
        "format_version": 1,
        "vocab_size": len(tokenizer.vocab),
        "requested_vocab_size": vocab_size,
        "dtype": dtype.name,
        "special_tokens": special_tokens,
        "train_text": str(Path(train_text).resolve()),
        "valid_text": str(Path(valid_text).resolve()),
        "train_tokens": train_token_count,
        "valid_tokens": valid_token_count,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--valid-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help=f"Repeat for multiple special tokens (default: {DEFAULT_SPECIAL_TOKEN})",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_tokens if args.special_tokens is not None else [DEFAULT_SPECIAL_TOKEN]
    metadata = prepare_dataset(
        args.train_text,
        args.valid_text,
        args.output_dir,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
