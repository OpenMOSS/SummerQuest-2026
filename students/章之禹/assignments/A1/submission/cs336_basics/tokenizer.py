"""Encoding and decoding for byte-level BPE tokenizers."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import regex

from cs336_basics.bpe import GPT2_PRETOKEN_PATTERN


Pair = tuple[bytes, bytes]
type Unit = tuple[Literal["normal", "special"], str, int, int]


class Tokenizer:
    """A deterministic GPT-2-style byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)

        specials: list[str] = []
        seen_specials: set[str] = set()
        for special in special_tokens or ():
            if not isinstance(special, str):
                raise TypeError("special tokens must be strings")
            if not special:
                raise ValueError("special tokens must not be empty")
            if special not in seen_specials:
                seen_specials.add(special)
                specials.append(special)

        value_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in sorted(self.vocab.items()):
            if not isinstance(token_id, int) or not isinstance(token_bytes, bytes):
                raise TypeError("vocab must map integer IDs to bytes")
            value_to_id.setdefault(token_bytes, token_id)

        next_id = max(self.vocab, default=-1) + 1
        self._special_to_id: dict[str, int] = {}
        for special in specials:
            encoded = special.encode("utf-8")
            token_id = value_to_id.get(encoded)
            if token_id is None:
                token_id = next_id
                next_id += 1
                self.vocab[token_id] = encoded
                value_to_id[encoded] = token_id
            self._special_to_id[special] = token_id

        self._bytes_to_id = value_to_id
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._specials = tuple(sorted(specials, key=lambda token: (-len(token), token)))
        self._max_special_length = max((len(token) for token in self._specials), default=0)
        self._special_pattern = (
            regex.compile("|".join(regex.escape(token) for token in self._specials)) if self._specials else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Construct a tokenizer from files written by ``save_tokenizer_files``."""

        vocab_payload = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges_payload = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        if vocab_payload.get("format") != "cs336-byte-bpe-v1":
            raise ValueError("unsupported vocabulary format")
        if merges_payload.get("format") != "cs336-byte-bpe-v1":
            raise ValueError("unsupported merges format")

        vocab = {
            int(item["id"]): base64.b64decode(item["bytes_base64"], validate=True) for item in vocab_payload["vocab"]
        }
        merges = [
            (base64.b64decode(left, validate=True), base64.b64decode(right, validate=True))
            for left, right in merges_payload["merges"]
        ]
        return cls(vocab, merges, special_tokens)

    def _scan_units(self, text: str) -> Iterator[Unit]:
        """Yield pre-token and special-token units together with source spans."""

        cursor = 0
        special_matches = () if self._special_pattern is None else self._special_pattern.finditer(text)
        for special_match in special_matches:
            for match in GPT2_PRETOKEN_PATTERN.finditer(text, cursor, special_match.start()):
                yield ("normal", match.group(), match.start(), match.end())
            special = special_match.group()
            yield ("special", special, special_match.start(), special_match.end())
            cursor = special_match.end()
        for match in GPT2_PRETOKEN_PATTERN.finditer(text, cursor, len(text)):
            yield ("normal", match.group(), match.start(), match.end())

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        symbols = [bytes((value,)) for value in pretoken.encode("utf-8")]
        while len(symbols) > 1:
            best_pair: Pair | None = None
            best_rank = len(self.merges)
            for left, right in zip(symbols, symbols[1:]):
                pair = (left, right)
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            merged_token = best_pair[0] + best_pair[1]
            new_symbols: list[bytes] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and symbols[index] == best_pair[0] and symbols[index + 1] == best_pair[1]:
                    new_symbols.append(merged_token)
                    index += 2
                else:
                    new_symbols.append(symbols[index])
                    index += 1
            symbols = new_symbols

        try:
            return [self._bytes_to_id[symbol] for symbol in symbols]
        except KeyError as error:
            raise ValueError(f"vocabulary has no token for byte sequence {error.args[0]!r}") from error

    def _encode_unit(self, kind: Literal["normal", "special"], value: str) -> Iterator[int]:
        if kind == "special":
            yield self._special_to_id[value]
        else:
            yield from self._encode_pretoken(value)

    def encode(self, text: str) -> list[int]:
        """Encode a Unicode string into token IDs."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        result: list[int] = []
        cache: dict[str, list[int]] = {}
        for kind, value, _, _ in self._scan_units(text):
            if kind == "special":
                result.append(self._special_to_id[value])
            else:
                ids = cache.get(value)
                if ids is None:
                    ids = self._encode_pretoken(value)
                    cache[value] = ids
                result.extend(ids)
        return result

    def _partial_special_start(self, text: str) -> int | None:
        """Return the earliest suffix that could grow into a special token."""

        if not self._specials or not text:
            return None
        lower_bound = max(0, len(text) - self._max_special_length)
        for start in range(lower_bound, len(text)):
            suffix = text[start:]
            if any(special.startswith(suffix) for special in self._specials):
                return start
        return None

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode chunks while preserving tokens across chunk boundaries."""

        pending = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("encode_iterable expects an iterable of strings")
            if not chunk:
                continue
            text = pending + chunk
            units = list(self._scan_units(text))
            if not units:
                pending = text
                continue

            # The final regex unit can change when the next chunk arrives. A
            # suffix that is a special-token prefix can begin even earlier.
            retain_start = units[-1][2]
            partial_start = self._partial_special_start(text)
            if partial_start is not None:
                partial_unit_index: int | None = None
                for index, (_, _, start, end) in enumerate(units):
                    if start <= partial_start < end:
                        partial_unit_index = index
                        retain_start = start
                        break

                # A suffix that is currently ordinary text can become a hard
                # special-token boundary after the next chunk arrives.  The
                # GPT-2 whitespace alternatives may then regroup the complete
                # whitespace run immediately before that boundary (for
                # example, ``"\n\nX"`` followed by ``"Y"`` when ``"XY"`` is
                # special).  Retain that run as well so streaming encoding is
                # identical to encoding the concatenated text.
                if partial_unit_index is not None and units[partial_unit_index][0] == "normal":
                    previous = partial_unit_index - 1
                    while previous >= 0 and units[previous][0] == "normal" and units[previous][1].isspace():
                        retain_start = units[previous][2]
                        previous -= 1

            for kind, value, _, end in units:
                if end > retain_start:
                    break
                yield from self._encode_unit(kind, value)
            pending = text[retain_start:]

        for kind, value, _, _ in self._scan_units(pending):
            yield from self._encode_unit(kind, value)

    def decode(self, ids: list[int]) -> str:
        """Decode IDs, replacing byte sequences that are not valid UTF-8."""

        try:
            encoded = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID {error.args[0]}") from error
        return encoded.decode("utf-8", errors="replace")
