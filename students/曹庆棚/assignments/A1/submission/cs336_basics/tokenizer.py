from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache
from pathlib import Path

import regex

from cs336_basics.bpe import GPT2_PRETOKEN_PATTERN


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        if len(self.bytes_to_id) != len(self.vocab):
            raise ValueError("vocab byte strings must be unique")
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.special_token_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_id = self.bytes_to_id.get(token.encode("utf-8"))
            if token_id is None:
                raise ValueError(f"special token {token!r} is absent from vocab")
            self.special_token_to_id[token] = token_id

        nonempty = sorted((token for token in self.special_tokens if token), key=len, reverse=True)
        self._special_pattern = regex.compile("|".join(regex.escape(token) for token in nonempty)) if nonempty else None
        self._cached_apply_merges: Callable[[bytes], tuple[bytes, ...]] | None = None

    def _apply_merges_uncached(self, token_bytes: bytes) -> tuple[bytes, ...]:
        pieces = [bytes([value]) for value in token_bytes]
        while len(pieces) >= 2:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for pair in zip(pieces, pieces[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_pieces: list[bytes] = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and pieces[index] == best_pair[0] and pieces[index + 1] == best_pair[1]:
                    new_pieces.append(merged)
                    index += 2
                else:
                    new_pieces.append(pieces[index])
                    index += 1
            pieces = new_pieces
        return tuple(pieces)

    def _apply_merges(self, token_bytes: bytes) -> tuple[bytes, ...]:
        if self._cached_apply_merges is not None:
            return self._cached_apply_merges(token_bytes)
        return self._apply_merges_uncached(token_bytes)

    def enable_merge_cache(self, max_size: int = 8192) -> None:
        """Cache BPE results for repeated pre-tokens during offline corpus encoding."""
        if max_size <= 0:
            self._cached_apply_merges = None
            return
        self._cached_apply_merges = lru_cache(maxsize=max_size)(self._apply_merges_uncached)

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            for token_bytes in self._apply_merges(match.group(0).encode("utf-8")):
                yield self.bytes_to_id[token_bytes]

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self._special_pattern is None:
            return list(self._encode_ordinary(text))

        result: list[int] = []
        previous_end = 0
        for match in self._special_pattern.finditer(text):
            result.extend(self._encode_ordinary(text[previous_end : match.start()]))
            result.append(self.special_token_to_id[match.group(0)])
            previous_end = match.end()
        result.extend(self._encode_ordinary(text[previous_end:]))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        try:
            encoded = b"".join(self.vocab[int(token_id)] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return encoded.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab": {
                str(token_id): base64.b64encode(token_bytes).decode("ascii")
                for token_id, token_bytes in self.vocab.items()
            },
            "merges": [
                [base64.b64encode(left).decode("ascii"), base64.b64encode(right).decode("ascii")]
                for left, right in sorted(self.merge_ranks, key=self.merge_ranks.get)
            ],
            "special_tokens": self.special_tokens,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {int(token_id): base64.b64decode(encoded) for token_id, encoded in payload["vocab"].items()}
        merges = [(base64.b64decode(left), base64.b64decode(right)) for left, right in payload["merges"]]
        return cls(vocab=vocab, merges=merges, special_tokens=payload.get("special_tokens"))
