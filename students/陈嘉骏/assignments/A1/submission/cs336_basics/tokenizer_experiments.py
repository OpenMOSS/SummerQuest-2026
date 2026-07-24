from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import BPETokenizer


TOKENIZER_ARTIFACT_VERSION = 1
DEFAULT_TEXT_CHUNK_SIZE = 1024 * 1024
DEFAULT_TOKEN_BATCH_SIZE = 65_536


@dataclass(frozen=True)
class TokenizerBenchmark:
    num_bytes: int
    num_tokens: int
    compression_ratio: float
    elapsed_seconds: float
    bytes_per_second: float
    tokens_per_second: float
    longest_observed_token_id: int | None
    longest_observed_token_num_bytes: int


@dataclass(frozen=True)
class EncodedDatasetInfo:
    dtype: str
    token_count: int
    vocab_size: int
    output_num_bytes: int


def save_tokenizer_artifact(
    output_path: str | os.PathLike[str],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> None:
    """Save a byte-level BPE tokenizer in a portable JSON representation."""
    artifact = {
        "version": TOKENIZER_ARTIFACT_VERSION,
        "vocab": [{"id": token_id, "bytes_hex": token_bytes.hex()} for token_id, token_bytes in sorted(vocab.items())],
        "merges": [[left.hex(), right.hex()] for left, right in merges],
        "special_tokens": [] if special_tokens is None else list(special_tokens),
    }
    _write_json_atomically(Path(output_path), artifact)


def load_tokenizer_artifact(input_path: str | os.PathLike[str]) -> BPETokenizer:
    """Load an artifact produced by :func:`save_tokenizer_artifact`."""
    with open(input_path, encoding="utf-8") as f:
        artifact = json.load(f)

    if artifact.get("version") != TOKENIZER_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported tokenizer artifact version: {artifact.get('version')!r}")

    vocab = _parse_vocab(artifact.get("vocab"))
    merges = _parse_merges(artifact.get("merges"))
    special_tokens = artifact.get("special_tokens")
    if not isinstance(special_tokens, list) or not all(isinstance(token, str) for token in special_tokens):
        raise ValueError("Tokenizer artifact special_tokens must be a list of strings.")

    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def train_tokenizer_artifact(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
) -> BPETokenizer:
    """Train a BPE tokenizer, persist it, and return the in-memory tokenizer."""
    resolved_special_tokens = [] if special_tokens is None else special_tokens
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=resolved_special_tokens,
        num_processes=num_processes,
    )
    save_tokenizer_artifact(output_path, vocab, merges, resolved_special_tokens)
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=resolved_special_tokens)


def iter_text_file_chunks(
    input_path: str | os.PathLike[str],
    chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE,
) -> Iterator[str]:
    """Yield bounded text chunks while preserving the file's newline characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    with open(input_path, encoding="utf-8", newline="") as f:
        while chunk := f.read(chunk_size):
            yield chunk


def benchmark_tokenizer(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike[str],
    chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE,
) -> TokenizerBenchmark:
    """Measure bytes/token and streaming encode throughput for one UTF-8 file."""
    num_bytes = os.path.getsize(input_path)
    num_tokens = 0
    longest_token_id: int | None = None
    longest_token_num_bytes = 0

    start_time = time.perf_counter()
    for token_id in tokenizer.encode_iterable(iter_text_file_chunks(input_path, chunk_size)):
        num_tokens += 1
        token_num_bytes = len(tokenizer.vocab[token_id])
        if token_num_bytes > longest_token_num_bytes:
            longest_token_id = token_id
            longest_token_num_bytes = token_num_bytes
    elapsed_seconds = time.perf_counter() - start_time

    compression_ratio = num_bytes / num_tokens if num_tokens else 0.0
    bytes_per_second = num_bytes / elapsed_seconds if elapsed_seconds else float("inf")
    tokens_per_second = num_tokens / elapsed_seconds if elapsed_seconds else float("inf")
    return TokenizerBenchmark(
        num_bytes=num_bytes,
        num_tokens=num_tokens,
        compression_ratio=compression_ratio,
        elapsed_seconds=elapsed_seconds,
        bytes_per_second=bytes_per_second,
        tokens_per_second=tokens_per_second,
        longest_observed_token_id=longest_token_id,
        longest_observed_token_num_bytes=longest_token_num_bytes,
    )


def choose_token_dtype(vocab: dict[int, bytes]) -> np.dtype[Any]:
    """Choose the smallest unsigned NumPy dtype that can represent every token ID."""
    if not vocab:
        return np.dtype(np.uint16)

    minimum_token_id = min(vocab)
    maximum_token_id = max(vocab)
    if minimum_token_id < 0:
        raise ValueError("Token IDs must be non-negative.")
    if maximum_token_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum_token_id <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    if maximum_token_id <= np.iinfo(np.uint64).max:
        return np.dtype(np.uint64)
    raise ValueError("Token IDs exceed the range supported by uint64.")


def encode_file_to_numpy_binary(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE,
    token_batch_size: int = DEFAULT_TOKEN_BATCH_SIZE,
    dtype: str | np.dtype[Any] | None = None,
) -> EncodedDatasetInfo:
    """Stream a UTF-8 corpus into a raw NumPy-compatible token ID file.

    A small ``<output>.json`` sidecar records the dtype and shape. The raw file can
    then be opened with :func:`load_encoded_dataset` without loading it into RAM.
    """
    if token_batch_size <= 0:
        raise ValueError("token_batch_size must be positive.")

    resolved_dtype = choose_token_dtype(tokenizer.vocab) if dtype is None else np.dtype(dtype)
    if resolved_dtype.kind != "u":
        raise ValueError("dtype must be an unsigned integer dtype.")
    if tokenizer.vocab and max(tokenizer.vocab) > np.iinfo(resolved_dtype).max:
        raise ValueError(f"dtype {resolved_dtype.name} cannot represent all token IDs.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    token_ids = tokenizer.encode_iterable(iter_text_file_chunks(input_path, chunk_size))
    token_count = 0

    try:
        with open(temporary_output, "wb") as f:
            while True:
                batch = np.fromiter(islice(token_ids, token_batch_size), dtype=resolved_dtype)
                if batch.size == 0:
                    break
                batch.tofile(f)
                token_count += int(batch.size)
        os.replace(temporary_output, output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    info = EncodedDatasetInfo(
        dtype=resolved_dtype.name,
        token_count=token_count,
        vocab_size=len(tokenizer.vocab),
        output_num_bytes=token_count * resolved_dtype.itemsize,
    )
    _write_json_atomically(_metadata_path(output), {"format": "numpy_raw", **asdict(info)})
    return info


def load_encoded_dataset(
    input_path: str | os.PathLike[str],
    mode: Literal["r", "r+", "c"] = "r",
) -> np.memmap[Any, Any]:
    """Memory-map a dataset produced by :func:`encode_file_to_numpy_binary`."""
    input_file = Path(input_path)
    with open(_metadata_path(input_file), encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("format") != "numpy_raw":
        raise ValueError(f"Unsupported encoded dataset format: {metadata.get('format')!r}")
    dtype = np.dtype(metadata["dtype"])
    token_count = metadata["token_count"]
    if not isinstance(token_count, int) or token_count < 0:
        raise ValueError("Encoded dataset token_count must be a non-negative integer.")

    expected_num_bytes = token_count * dtype.itemsize
    actual_num_bytes = input_file.stat().st_size
    if actual_num_bytes != expected_num_bytes:
        raise ValueError(
            f"Encoded dataset size mismatch: expected {expected_num_bytes} bytes, found {actual_num_bytes}."
        )
    return np.memmap(input_file, dtype=dtype, mode=mode, shape=(token_count,))


def longest_vocab_tokens(tokenizer: BPETokenizer, limit: int = 10) -> list[tuple[int, bytes]]:
    """Return the longest vocabulary entries, with deterministic tie-breaking."""
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    return sorted(tokenizer.vocab.items(), key=lambda item: (len(item[1]), item[1], item[0]), reverse=True)[:limit]


def _parse_vocab(value: object) -> dict[int, bytes]:
    if not isinstance(value, list):
        raise ValueError("Tokenizer artifact vocab must be a list.")

    vocab: dict[int, bytes] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Each vocab entry must be an object.")
        typed_entry = cast(dict[str, object], entry)
        token_id = typed_entry.get("id")
        bytes_hex = typed_entry.get("bytes_hex")
        if not isinstance(token_id, int) or token_id < 0 or not isinstance(bytes_hex, str):
            raise ValueError("Invalid tokenizer vocab entry.")
        if token_id in vocab:
            raise ValueError(f"Duplicate token ID in tokenizer artifact: {token_id}")
        try:
            vocab[token_id] = bytes.fromhex(bytes_hex)
        except ValueError as error:
            raise ValueError(f"Invalid hex bytes for token ID {token_id}.") from error
    return vocab


def _parse_merges(value: object) -> list[tuple[bytes, bytes]]:
    if not isinstance(value, list):
        raise ValueError("Tokenizer artifact merges must be a list.")

    merges: list[tuple[bytes, bytes]] = []
    for entry in value:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("Each merge entry must contain two hex strings.")
        left_hex, right_hex = entry
        if not isinstance(left_hex, str) or not isinstance(right_hex, str):
            raise ValueError("Each merge entry must contain two hex strings.")
        try:
            merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
        except ValueError as error:
            raise ValueError("Invalid hex bytes in tokenizer merge entry.") from error
    return merges


def _metadata_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(dataset_path.name + ".json")


def _write_json_atomically(output_path: Path, value: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(temporary_output, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary_output, output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
