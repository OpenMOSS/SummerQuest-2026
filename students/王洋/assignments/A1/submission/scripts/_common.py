"""Shared helpers for the reproducible command-line scripts.

The tokenizer artifact intentionally uses GPT-2's printable byte-to-Unicode
representation so that it can be loaded directly by ``Tokenizer.from_files``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import psutil
import torch

from cs336_basics.tokenizer import Tokenizer


TOKENIZER_METADATA_FILENAME = "tokenizer.json"
TOKENIZER_FORMAT = "cs336-byte-bpe-v1"


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a sibling temporary file and an atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _gpt2_byte_encoder() -> dict[int, str]:
    """Return GPT-2's reversible mapping from all byte values to Unicode."""

    visible = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = visible[:]
    unicode_values = visible[:]
    offset = 0
    for byte in range(256):
        if byte not in visible:
            byte_values.append(byte)
            unicode_values.append(256 + offset)
            offset += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(byte_values, unicode_values, strict=True)}


def _serialize_token(token: bytes, encoder: Mapping[int, str]) -> str:
    return "".join(encoder[byte] for byte in token)


def save_tokenizer_artifact(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    training_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a BPE vocabulary, merges, and self-describing metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    metadata_path = output_dir / TOKENIZER_METADATA_FILENAME
    encoder = _gpt2_byte_encoder()

    serialized_vocab: dict[str, int] = {}
    for token_id, token in sorted(vocab.items()):
        serialized = _serialize_token(token, encoder)
        if serialized in serialized_vocab:
            raise ValueError(f"vocabulary has duplicate byte sequence at token ID {token_id}")
        serialized_vocab[serialized] = token_id
    atomic_write_json(vocab_path, serialized_vocab)

    temporary_merges = merges_path.with_name(f".{merges_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_merges.open("w", encoding="utf-8") as merges_file:
            merges_file.write("#version: 0.2\n")
            for left, right in merges:
                merges_file.write(f"{_serialize_token(left, encoder)} {_serialize_token(right, encoder)}\n")
            merges_file.flush()
            os.fsync(merges_file.fileno())
        os.replace(temporary_merges, merges_path)
    finally:
        temporary_merges.unlink(missing_ok=True)

    metadata: dict[str, Any] = {
        "format": TOKENIZER_FORMAT,
        "vocab_file": vocab_path.name,
        "merges_file": merges_path.name,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": special_tokens,
        "sha256": {
            "vocab_file": sha256_file(vocab_path),
            "merges_file": sha256_file(merges_path),
        },
    }
    if training_summary is not None:
        metadata["training"] = training_summary
    atomic_write_json(metadata_path, metadata)
    return metadata


def load_tokenizer_artifact(path: str | os.PathLike[str]) -> tuple[Tokenizer, dict[str, Any], Path]:
    """Load a tokenizer directory or its ``tokenizer.json`` metadata file."""

    artifact_path = Path(path)
    metadata_path = artifact_path / TOKENIZER_METADATA_FILENAME if artifact_path.is_dir() else artifact_path
    metadata = load_json(metadata_path)
    if metadata.get("format") != TOKENIZER_FORMAT:
        raise ValueError(
            f"unsupported tokenizer artifact format {metadata.get('format')!r}; expected {TOKENIZER_FORMAT!r}"
        )

    artifact_dir = metadata_path.parent
    vocab_path = artifact_dir / str(metadata.get("vocab_file", "vocab.json"))
    merges_path = artifact_dir / str(metadata.get("merges_file", "merges.txt"))
    special_tokens = metadata.get("special_tokens", [])
    if not isinstance(special_tokens, list) or not all(isinstance(token, str) for token in special_tokens):
        raise ValueError("tokenizer metadata special_tokens must be a list of strings")

    expected_hashes = metadata.get("sha256", {})
    if isinstance(expected_hashes, dict):
        for key, file_path in (("vocab_file", vocab_path), ("merges_file", merges_path)):
            expected = expected_hashes.get(key)
            if expected is not None and sha256_file(file_path) != expected:
                raise ValueError(f"checksum mismatch for tokenizer artifact file {file_path.name}")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    if len(tokenizer.vocab) != metadata.get("vocab_size"):
        raise ValueError("tokenizer vocabulary size does not match its metadata")
    return tokenizer, metadata, metadata_path


def iter_text_chunks(
    path: str | os.PathLike[str],
    chunk_chars: int,
    *,
    max_chars: int | None = None,
    errors: str = "strict",
) -> Iterator[str]:
    """Yield bounded text chunks without loading a complete corpus into RAM."""

    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    remaining = max_chars
    with Path(path).open(encoding="utf-8", errors=errors) as input_file:
        while remaining is None or remaining > 0:
            requested = chunk_chars if remaining is None else min(chunk_chars, remaining)
            chunk = input_file.read(requested)
            if not chunk:
                break
            yield chunk
            if remaining is not None:
                remaining -= len(chunk)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and (getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return device


class PeakRSSMonitor:
    """Sample resident memory in a background thread during a bounded operation."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PeakRSSMonitor:
        process = psutil.Process()
        self.peak_bytes = max(process.memory_info().rss, _linux_high_water_rss_bytes())

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
                except psutil.Error:
                    return

        self._thread = threading.Thread(target=sample, name="peak-rss-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        try:
            self.peak_bytes = max(
                self.peak_bytes,
                psutil.Process().memory_info().rss,
                _linux_high_water_rss_bytes(),
            )
        except psutil.Error:
            self.peak_bytes = max(self.peak_bytes, _linux_high_water_rss_bytes())


def _linux_high_water_rss_bytes() -> int:
    """Read the kernel-recorded process RSS high-water mark when available.

    Sampling RSS from a Python thread can miss brief allocation spikes,
    especially while another thread holds the GIL.  Linux records ``VmHWM``
    independently, so use it as the authoritative lower bound for resource
    reporting and retain the sampled fallback on other platforms.
    """

    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
