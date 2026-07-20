from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import regex as re

from cs336_basics.bpe import GPT2_PRETOKENIZATION_PATTERN


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self._token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        if len(self._token_to_id) != len(self.vocab):
            raise ValueError("Vocabulary bytes must map one-to-one to token IDs.")
        self._merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._pretoken_pattern = re.compile(GPT2_PRETOKENIZATION_PATTERN)

        # Natural-language corpora repeat many pretokens. A bounded cache avoids
        # re-running the same BPE merge sequence without growing with the corpus.
        self._encode_pretoken_cached = lru_cache(maxsize=512)(self._encode_pretoken_uncached)

        unique_special_tokens: list[str] = [] if special_tokens is None else list(dict.fromkeys(special_tokens))
        for token in unique_special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self._token_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = token_bytes
                self._token_to_id[token_bytes] = token_id

        self.special_tokens: list[str] = sorted(
            unique_special_tokens,
            key=lambda token: len(token),
            reverse=True,
        )
        self._special_token_to_id = {token: self._token_to_id[token.encode("utf-8")] for token in self.special_tokens}
        self._max_special_token_length = max((len(token) for token in self.special_tokens), default=0)
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(token) for token in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Construct a tokenizer from the JSON files produced by :meth:`to_files`."""
        with open(vocab_filepath, encoding="utf-8") as f:
            serialized_vocab = json.load(f)
        with open(merges_filepath, encoding="utf-8") as f:
            serialized_merges = json.load(f)
        return cls(
            vocab=_parse_serialized_vocab(serialized_vocab),
            merges=_parse_serialized_merges(serialized_merges),
            special_tokens=special_tokens,
        )

    def to_files(
        self,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
    ) -> None:
        """Serialize vocabulary and merges as separate JSON files."""
        serialized_vocab = {str(token_id): token_bytes.hex() for token_id, token_bytes in sorted(self.vocab.items())}
        serialized_merges = [[left.hex(), right.hex()] for left, right in self.merges]
        _write_json_atomically(Path(vocab_filepath), serialized_vocab)
        _write_json_atomically(Path(merges_filepath), serialized_merges)

    def encode(self, text: str) -> list[int]:
        return list(self._encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            if chunk == "":
                continue
            buffer += chunk
            ids, buffer = self._encode_stream_buffer(buffer, final=False)
            yield from ids

        ids, buffer = self._encode_stream_buffer(buffer, final=True)
        yield from ids
        if buffer:
            raise RuntimeError("Tokenizer stream ended with unprocessed text.")

    def decode(self, ids: Iterable[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def _encode_text(self, text: str) -> Iterator[int]:
        if self._special_pattern is None:
            yield from self._encode_ordinary_text(text)
            return

        cursor = 0
        for match in self._special_pattern.finditer(text):
            yield from self._encode_ordinary_text(text[cursor : match.start()])
            yield self._special_token_to_id[match.group(0)]
            cursor = match.end()
        yield from self._encode_ordinary_text(text[cursor:])

    def _encode_ordinary_text(self, text: str) -> Iterator[int]:
        for match in self._pretoken_pattern.finditer(text):
            yield from self._encode_pretoken(match.group(0).encode("utf-8"))

    def _encode_pretoken(self, pretoken: bytes) -> Iterator[int]:
        if not pretoken:
            return

        yield from self._encode_pretoken_cached(pretoken)

    def _encode_pretoken_uncached(self, pretoken: bytes) -> tuple[int, ...]:
        """Encode one pretoken using only the learned merge ranking."""

        tokens = tuple(bytes([byte]) for byte in pretoken)
        while len(tokens) > 1:
            best_pair = min(
                zip(tokens, tokens[1:]),
                key=lambda pair: self._merge_ranks.get(pair, float("inf")),
            )
            if best_pair not in self._merge_ranks:
                break
            tokens = self._merge_bytes_token_sequence(tokens, best_pair)

        return tuple(self._token_to_id[token] for token in tokens)

    @staticmethod
    def _merge_bytes_token_sequence(
        tokens: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        merged_token = pair[0] + pair[1]
        merged: list[bytes] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(merged_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return tuple(merged)

    def _encode_stream_buffer(self, buffer: str, final: bool) -> tuple[list[int], str]:
        ids: list[int] = []

        while buffer:
            special_match = self._find_safe_special_token(buffer, final)
            if special_match is not None:
                ids.extend(self._encode_ordinary_text(buffer[: special_match.start()]))
                ids.append(self._special_token_to_id[special_match.group(0)])
                buffer = buffer[special_match.end() :]
                continue

            end_limit = self._ordinary_stream_end_limit(buffer, final)
            ordinary_ids, consumed = self._encode_ordinary_prefix(buffer, end_limit, final)
            ids.extend(ordinary_ids)
            if consumed == 0:
                break
            buffer = buffer[consumed:]

        return ids, buffer

    def _find_safe_special_token(self, buffer: str, final: bool) -> Any | None:
        if self._special_pattern is None:
            return None

        match = self._special_pattern.search(buffer)
        if match is None:
            return None

        if final:
            return match

        safe_start = len(buffer) - self._max_special_token_length
        return match if match.start() <= safe_start else None

    def _ordinary_stream_end_limit(self, buffer: str, final: bool) -> int:
        if final or not self.special_tokens:
            return len(buffer)
        return max(0, len(buffer) - self._max_special_token_length + 1)

    def _encode_ordinary_prefix(
        self,
        buffer: str,
        end_limit: int,
        final: bool,
    ) -> tuple[list[int], int]:
        ids: list[int] = []
        consumed = 0

        for match in self._pretoken_pattern.finditer(buffer):
            if match.end() > end_limit:
                break
            if not final and match.end() == len(buffer):
                break

            ids.extend(self._encode_pretoken(match.group(0).encode("utf-8")))
            consumed = match.end()

        return ids, consumed


def _parse_serialized_vocab(value: object) -> dict[int, bytes]:
    if not isinstance(value, dict):
        raise ValueError("Serialized vocabulary must be a JSON object mapping IDs to hex bytes.")

    vocab: dict[int, bytes] = {}
    for raw_token_id, raw_token_bytes in value.items():
        if not isinstance(raw_token_id, str) or not isinstance(raw_token_bytes, str):
            raise ValueError("Serialized vocabulary entries must map string IDs to hex strings.")
        try:
            token_id = int(raw_token_id)
            token_bytes = bytes.fromhex(raw_token_bytes)
        except ValueError as error:
            raise ValueError("Invalid token ID or hex bytes in serialized vocabulary.") from error
        if token_id < 0 or token_id in vocab:
            raise ValueError(f"Invalid or duplicate token ID: {token_id}")
        vocab[token_id] = token_bytes
    return vocab


def _parse_serialized_merges(value: object) -> list[tuple[bytes, bytes]]:
    if not isinstance(value, list):
        raise ValueError("Serialized merges must be a JSON list.")

    merges: list[tuple[bytes, bytes]] = []
    for entry in value:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("Each serialized merge must contain two hex strings.")
        left, right = entry
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("Each serialized merge must contain two hex strings.")
        try:
            merges.append((bytes.fromhex(left), bytes.fromhex(right)))
        except ValueError as error:
            raise ValueError("Invalid hex bytes in serialized merge.") from error
    return merges


def _write_json_atomically(output_path: Path, value: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
