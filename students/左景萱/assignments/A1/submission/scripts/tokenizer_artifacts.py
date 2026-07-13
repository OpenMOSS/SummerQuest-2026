"""Shared helpers for tokenizer artifacts and bounded corpus streaming."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_special_tokens(values: list[str] | None, no_special_tokens: bool) -> list[str]:
    """Resolve the shared CLI convention for special tokens."""

    if no_special_tokens:
        if values:
            raise ValueError("--no-special-tokens cannot be combined with --special-token")
        return []
    tokens = values if values is not None else [DEFAULT_SPECIAL_TOKEN]
    if any(not token for token in tokens):
        raise ValueError("special tokens must be non-empty")
    return list(dict.fromkeys(tokens))


def longest_token_summary(vocab: dict[int, bytes], special_tokens: list[str]) -> dict[str, Any]:
    """Describe the longest vocabulary token, both with and without specials."""

    special_bytes = {token.encode("utf-8") for token in special_tokens}

    def describe(candidates: list[tuple[int, bytes]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        token_id, token = max(candidates, key=lambda item: (len(item[1]), -item[0]))
        return {
            "id": token_id,
            "length_bytes": len(token),
            "bytes_hex": token.hex(),
            "text_utf8": token.decode("utf-8", errors="replace"),
        }

    items = list(vocab.items())
    return {
        "including_special_tokens": describe(items),
        "excluding_special_tokens": describe(
            [(token_id, token) for token_id, token in items if token not in special_bytes]
        ),
    }


class CorpusTextStream:
    """Yield bounded UTF-8 text chunks while retaining stream statistics.

    Chunks preferentially end after a configured document delimiter. If a
    document is larger than ``chunk_bytes``, a newline is preferred, followed
    by a UTF-8-safe fixed boundary. This keeps memory bounded without dropping
    corpus bytes. A byte-limited sample that ends inside a UTF-8 codepoint omits
    only that incomplete codepoint and reports the omission.
    """

    def __init__(
        self,
        path: Path,
        *,
        chunk_bytes: int,
        max_bytes: int | None,
        document_delimiter: str | None,
    ) -> None:
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if not path.is_file():
            raise FileNotFoundError(f"corpus not found: {path}")

        self.path = path
        self.chunk_bytes = chunk_bytes
        self.max_bytes = max_bytes
        self.document_delimiter = document_delimiter
        self.source_bytes = path.stat().st_size
        self.bytes_read = 0
        self.bytes_processed = 0
        self.chunks = 0
        self.incomplete_utf8_tail_bytes = 0
        self._iterated = False

    @property
    def was_limited(self) -> bool:
        return self.max_bytes is not None and self.max_bytes < self.source_bytes

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_bytes": self.source_bytes,
            "sample_limit_bytes": self.max_bytes,
            "sample_was_limited": self.was_limited,
            "bytes_read": self.bytes_read,
            "bytes_processed": self.bytes_processed,
            "incomplete_utf8_tail_bytes": self.incomplete_utf8_tail_bytes,
            "chunks": self.chunks,
            "chunk_bytes_target": self.chunk_bytes,
            "document_delimiter": self.document_delimiter,
            "boundary_policy": "document_delimiter_then_newline_then_utf8_safe_fixed_boundary",
        }

    @staticmethod
    def _utf8_safe_prefix_length(buffer: bytearray, target: int) -> int:
        end = min(target, len(buffer))
        while end > max(0, target - 4):
            try:
                bytes(buffer[:end]).decode("utf-8")
                return end
            except UnicodeDecodeError as error:
                if error.end != end:
                    raise
                end = error.start
        raise UnicodeDecodeError("utf-8", bytes(buffer), 0, min(len(buffer), target), "no safe chunk boundary")

    def _split_position(self, buffer: bytearray) -> int:
        target = min(self.chunk_bytes, len(buffer))
        if self.document_delimiter:
            delimiter = self.document_delimiter.encode("utf-8")
            position = buffer.rfind(delimiter, 0, target + 1)
            if position >= 0:
                return position + len(delimiter)
        newline = buffer.rfind(b"\n", 0, target + 1)
        if newline >= 0:
            return newline + 1
        return self._utf8_safe_prefix_length(buffer, target)

    def _emit(self, buffer: bytearray, end: int) -> str:
        raw = bytes(buffer[:end])
        del buffer[:end]
        text = raw.decode("utf-8")
        self.bytes_processed += len(raw)
        self.chunks += 1
        return text

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("CorpusTextStream can only be iterated once")
        self._iterated = True

        budget = self.source_bytes if self.max_bytes is None else min(self.source_bytes, self.max_bytes)
        buffer = bytearray()
        with self.path.open("rb") as file:
            while self.bytes_read < budget:
                raw = file.read(min(self.chunk_bytes, budget - self.bytes_read))
                if not raw:
                    break
                self.bytes_read += len(raw)
                buffer.extend(raw)
                while len(buffer) >= self.chunk_bytes:
                    yield self._emit(buffer, self._split_position(buffer))

        if buffer:
            if self.was_limited:
                try:
                    end = len(buffer)
                    bytes(buffer).decode("utf-8")
                except UnicodeDecodeError as error:
                    if error.end != len(buffer):
                        raise
                    end = error.start
                    self.incomplete_utf8_tail_bytes = len(buffer) - end
                if end:
                    yield self._emit(buffer, end)
            else:
                yield self._emit(buffer, len(buffer))


__all__ = [
    "CorpusTextStream",
    "DEFAULT_SPECIAL_TOKEN",
    "longest_token_summary",
    "resolve_special_tokens",
    "utc_now",
    "write_json_atomic",
]
