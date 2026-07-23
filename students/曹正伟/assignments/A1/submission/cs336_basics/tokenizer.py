"""A byte-level BPE tokenizer with GPT-2-compatible pre-tokenization."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import regex

from .bpe import PAT


# Reuse these immutable objects rather than allocating one bytes object for
# every byte of every cache miss.
_SINGLE_BYTE_TOKENS = tuple(
    bytes((byte_value,))
    for byte_value in range(256)
)


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse of GPT-2's printable byte-to-Unicode mapping."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    code_points = byte_values[:]
    offset = 0

    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(256 + offset)
            offset += 1

    return {
        chr(code_point): byte_value
        for byte_value, code_point in zip(byte_values, code_points)
    }


def _decode_gpt2_token(
    token: str,
    decoder: dict[str, int],
) -> bytes:
    try:
        return bytes(
            decoder[character]
            for character in token
        )
    except KeyError as error:
        raise ValueError(
            f"Invalid GPT-2 byte-unicode token {token!r}"
        ) from error


def _load_vocab(
    path: str | Path,
) -> dict[int, bytes]:
    with Path(path).open(encoding="utf-8") as vocab_file:
        raw_vocab: Any = json.load(vocab_file)

    if not isinstance(raw_vocab, dict):
        raise ValueError("Vocabulary JSON must contain an object")

    # Portable project format:
    #
    # {
    #     "0": "00",
    #     "1": "ff"
    # }
    #
    # Values are bytes.hex() strings.
    if all(
        isinstance(value, str)
        for value in raw_vocab.values()
    ):
        try:
            return {
                int(token_id): bytes.fromhex(encoded)
                for token_id, encoded in raw_vocab.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Invalid portable id-to-hex vocabulary JSON"
            ) from error

    # GPT-2 format:
    #
    # {
    #     printable_byte_unicode_token: integer_id
    # }
    if all(
        isinstance(value, int)
        and not isinstance(value, bool)
        for value in raw_vocab.values()
    ):
        decoder = _gpt2_byte_decoder()

        return {
            token_id: _decode_gpt2_token(token, decoder)
            for token, token_id in raw_vocab.items()
        }

    raise ValueError("Unsupported vocabulary JSON format")


def _load_merges(
    path: str | Path,
) -> list[tuple[bytes, bytes]]:
    merge_path = Path(path)

    try:
        with merge_path.open(encoding="utf-8") as merge_file:
            raw_merges: Any = json.load(merge_file)
    except json.JSONDecodeError:
        raw_merges = None

    # Portable project format:
    #
    # [
    #     [left.hex(), right.hex()],
    #     ...
    # ]
    if raw_merges is not None:
        if not isinstance(raw_merges, list):
            raise ValueError("Merges JSON must contain a list")

        result: list[tuple[bytes, bytes]] = []

        try:
            for item in raw_merges:
                if not isinstance(item, list | tuple) or len(item) != 2:
                    raise ValueError

                left, right = item

                if not isinstance(left, str) or not isinstance(right, str):
                    raise ValueError

                result.append(
                    (
                        bytes.fromhex(left),
                        bytes.fromhex(right),
                    )
                )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Invalid portable hex merges JSON"
            ) from error

        return result

    # GPT-2 whitespace-delimited merges format.
    decoder = _gpt2_byte_decoder()
    result = []

    with merge_path.open(encoding="utf-8") as merge_file:
        for line in merge_file:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            pieces = stripped.split()

            if len(pieces) != 2:
                continue

            result.append(
                (
                    _decode_gpt2_token(pieces[0], decoder),
                    _decode_gpt2_token(pieces[1], decoder),
                )
            )

    return result


class Tokenizer:
    """Encode and decode text using a byte-level BPE vocabulary."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        *,
        pretoken_cache_capacity: int = 16_384,
    ) -> None:
        if pretoken_cache_capacity < 0:
            raise ValueError(
                "pretoken_cache_capacity must be non-negative"
            )

        self.vocab = {
            int(token_id): bytes(token)
            for token_id, token in vocab.items()
        }

        self.merges = [
            (bytes(left), bytes(right))
            for left, right in merges
        ]

        self._token_to_id: dict[bytes, int] = {}

        for token_id, token in self.vocab.items():
            if (
                token in self._token_to_id
                and self._token_to_id[token] != token_id
            ):
                raise ValueError(
                    f"Vocabulary contains duplicate byte token {token!r}"
                )

            self._token_to_id[token] = token_id

        specials: list[str] = []
        seen_specials: set[str] = set()

        for special_token in special_tokens or []:
            if not special_token:
                raise ValueError(
                    "Special tokens must not be empty strings"
                )

            if special_token in seen_specials:
                continue

            seen_specials.add(special_token)
            specials.append(special_token)

            encoded = special_token.encode("utf-8")

            if encoded not in self._token_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = encoded
                self._token_to_id[encoded] = token_id

        self.special_tokens = specials

        self._special_to_id = {
            special_token: self._token_to_id[
                special_token.encode("utf-8")
            ]
            for special_token in specials
        }

        self.inverse_vocab = dict(self._token_to_id)

        if specials:
            longest_first = sorted(
                specials,
                key=len,
                reverse=True,
            )

            self._special_pattern = regex.compile(
                "|".join(
                    regex.escape(token)
                    for token in longest_first
                )
            )
        else:
            self._special_pattern = None

        # If a pair appears more than once in merges, its first occurrence has
        # the highest priority.
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {}

        for rank, pair in enumerate(self.merges):
            self._merge_ranks.setdefault(pair, rank)

        # The cache belongs to this Tokenizer instance rather than the class.
        # multiprocessing workers construct separate Tokenizer instances and
        # therefore automatically receive independent caches.
        self._pretoken_cache_capacity = pretoken_cache_capacity
        self._pretoken_cache: OrderedDict[
            bytes,
            tuple[int, ...],
        ] = OrderedDict()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
        *,
        pretoken_cache_capacity: int = 16_384,
    ) -> Tokenizer:
        """Construct a tokenizer from portable JSON or GPT-2 files."""
        return cls(
            vocab=_load_vocab(vocab_filepath),
            merges=_load_merges(merges_filepath),
            special_tokens=special_tokens,
            pretoken_cache_capacity=pretoken_cache_capacity,
        )

    def _encode_pretoken(
        self,
        pretoken: bytes,
    ) -> list[int]:
        """Encode one ordinary pre-token using the instance-local LRU cache."""
        if not pretoken:
            return []

        if self._pretoken_cache_capacity:
            cached = self._pretoken_cache.get(pretoken)

            if cached is not None:
                self._pretoken_cache.move_to_end(pretoken)

                # Preserve the old private method's list return type. The value
                # held by the cache itself remains an immutable tuple.
                return list(cached)

        parts = [
            _SINGLE_BYTE_TOKENS[byte_value]
            for byte_value in pretoken
        ]

        merge_ranks = self._merge_ranks

        while len(parts) > 1:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            for index in range(len(parts) - 1):
                pair = (
                    parts[index],
                    parts[index + 1],
                )
                rank = merge_ranks.get(pair)

                if (
                    rank is not None
                    and (
                        best_rank is None
                        or rank < best_rank
                    )
                ):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            left, right = best_pair
            merged = left + right

            new_parts: list[bytes] = []
            index = 0

            while index < len(parts):
                if (
                    index + 1 < len(parts)
                    and parts[index] == left
                    and parts[index + 1] == right
                ):
                    new_parts.append(merged)
                    index += 2
                else:
                    new_parts.append(parts[index])
                    index += 1

            parts = new_parts

        try:
            token_ids = tuple(
                self._token_to_id[part]
                for part in parts
            )
        except KeyError as error:
            raise ValueError(
                "BPE produced a token absent from the vocabulary: "
                f"{error.args[0]!r}"
            ) from error

        if self._pretoken_cache_capacity:
            # Only ordinary regex pre-tokens reach this method. Special tokens
            # are emitted directly by encode() and encode_iterable(), so they
            # never enter this cache.
            self._pretoken_cache[pretoken] = token_ids
            self._pretoken_cache.move_to_end(pretoken)

            if (
                len(self._pretoken_cache)
                > self._pretoken_cache_capacity
            ):
                self._pretoken_cache.popitem(last=False)

        return list(token_ids)

    def _encode_ordinary(
        self,
        text: str,
    ) -> Iterator[int]:
        for match in PAT.finditer(text):
            yield from self._encode_pretoken(
                match.group().encode("utf-8")
            )

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """Encode a Unicode string into token IDs."""
        if self._special_pattern is None:
            return list(self._encode_ordinary(text))

        token_ids: list[int] = []
        segment_start = 0

        for match in self._special_pattern.finditer(text):
            token_ids.extend(
                self._encode_ordinary(
                    text[segment_start : match.start()]
                )
            )

            token_ids.append(
                self._special_to_id[match.group()]
            )

            segment_start = match.end()

        token_ids.extend(
            self._encode_ordinary(
                text[segment_start:]
            )
        )

        return token_ids

    def _token_spans(
        self,
        text: str,
    ) -> list[tuple[int, int, str | None]]:
        """Split text into ordinary pre-token and special-token spans."""
        spans: list[tuple[int, int, str | None]] = []

        def add_ordinary(
            start: int,
            end: int,
        ) -> None:
            for match in PAT.finditer(text, start, end):
                spans.append(
                    (
                        match.start(),
                        match.end(),
                        None,
                    )
                )

        if self._special_pattern is None:
            add_ordinary(0, len(text))
            return spans

        segment_start = 0

        for match in self._special_pattern.finditer(text):
            add_ordinary(
                segment_start,
                match.start(),
            )

            spans.append(
                (
                    match.start(),
                    match.end(),
                    match.group(),
                )
            )

            segment_start = match.end()

        add_ordinary(
            segment_start,
            len(text),
        )

        return spans

    def _stable_prefix_length(
        self,
        text: str,
        spans: list[tuple[int, int, str | None]],
    ) -> int:
        """Return a prefix boundary whose tokenization cannot change later."""
        if not text:
            return 0

        cutoff = (
            spans[-1][0]
            if spans and spans[-1][1] == len(text)
            else len(text)
        )

        # Retain a suffix that is a prefix of any special token. This also
        # handles overlapping special tokens where a shorter token is a prefix
        # of a longer token.
        for special_token in self.special_tokens:
            max_prefix_length = min(
                len(text),
                len(special_token),
            )

            for prefix_length in range(
                1,
                max_prefix_length + 1,
            ):
                if text.endswith(
                    special_token[:prefix_length]
                ):
                    cutoff = min(
                        cutoff,
                        len(text) - prefix_length,
                    )

        return cutoff

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """Encode chunks exactly like encoding their concatenation."""
        pending = ""

        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError(
                    "Tokenizer.encode_iterable expects "
                    "an iterable of strings"
                )

            pending += chunk
            spans = self._token_spans(pending)

            stable_length = self._stable_prefix_length(
                pending,
                spans,
            )

            if stable_length:
                # Encode against the complete pending buffer. This preserves
                # GPT-2 PAT's trailing-whitespace lookahead behavior.
                for start, end, special_token in spans:
                    if end > stable_length:
                        break

                    if special_token is None:
                        yield from self._encode_pretoken(
                            pending[start:end].encode("utf-8")
                        )
                    else:
                        yield self._special_to_id[
                            special_token
                        ]

                pending = pending[stable_length:]

        if pending:
            yield from self.encode(pending)

    def decode(
        self,
        ids: Iterable[int],
    ) -> str:
        """Decode token IDs, replacing malformed UTF-8 with U+FFFD."""
        try:
            encoded = b"".join(
                self.vocab[int(token_id)]
                for token_id in ids
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown token ID: {error.args[0]}"
            ) from error

        return encoded.decode(
            "utf-8",
            errors="replace",
        )


__all__ = ["Tokenizer"]