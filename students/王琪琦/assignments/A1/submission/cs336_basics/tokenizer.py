from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex

from cs336_basics.bpe import GPT2_PRETOKEN_PATTERN


class Tokenizer:
    """A byte-level BPE tokenizer with optional indivisible special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        self._token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        missing_byte_tokens = [value for value in range(256) if bytes([value]) not in self._token_to_id]
        if missing_byte_tokens:
            raise ValueError("vocab must contain all 256 single-byte tokens")

        self._special_to_id: dict[str, int] = {}
        for special_token in self.special_tokens:
            token_id = self._token_to_id.get(special_token.encode("utf-8"))
            if token_id is None:
                raise ValueError(f"special token is not present in vocab: {special_token!r}")
            self._special_to_id[special_token] = token_id

        if self.special_tokens:
            longest_first = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = regex.compile(
                "|".join(regex.escape(token) for token in longest_first)
            )
        else:
            self._special_pattern = None

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = [bytes([byte_value]) for byte_value in pretoken.encode("utf-8")]

        while len(tokens) > 1:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank = len(self.merges)
            for pair in zip(tokens, tokens[1:]):
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            merged_tokens: list[bytes] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    merged_tokens.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            tokens = merged_tokens

        return [self._token_to_id[token] for token in tokens]

    def _encode_ordinary_text(self, text: str) -> Iterator[int]:
        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group())

    def encode(self, text: str) -> list[int]:
        """Encode text, preserving configured special tokens as single IDs."""
        if self._special_pattern is None:
            return list(self._encode_ordinary_text(text))

        token_ids: list[int] = []
        ordinary_start = 0
        for match in self._special_pattern.finditer(text):
            token_ids.extend(self._encode_ordinary_text(text[ordinary_start : match.start()]))
            token_ids.append(self._special_to_id[match.group()])
            ordinary_start = match.end()
        token_ids.extend(self._encode_ordinary_text(text[ordinary_start:]))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable without retaining earlier input chunks."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode IDs after joining bytes so UTF-8 characters may span tokens."""
        return b"".join(self.vocab[token_id] for token_id in ids).decode(
            "utf-8", errors="replace"
        )

    def special_token_id(self, token: str) -> int | None:
        return self._special_to_id.get(token)
