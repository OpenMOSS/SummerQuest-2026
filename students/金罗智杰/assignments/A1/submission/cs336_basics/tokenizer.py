from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex

from cs336_basics.bpe import BYTE_TOKENS, GPT2_PRETOKEN_PATTERN, _merge_pair


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        unique_specials = list(dict.fromkeys(special_tokens or []))
        self.special_to_id: dict[str, int] = {}
        for special in unique_specials:
            encoded = special.encode("utf-8")
            if encoded not in self.token_to_id:
                token_id = len(self.vocab)
                self.vocab[token_id] = encoded
                self.token_to_id[encoded] = token_id
            self.special_to_id[special] = self.token_to_id[encoded]

        if unique_specials:
            alternatives = sorted(unique_specials, key=len, reverse=True)
            self.special_pattern = regex.compile(
                "(" + "|".join(regex.escape(token) for token in alternatives) + ")"
            )
        else:
            self.special_pattern = None

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        encoded = pretoken.encode("utf-8")
        tokens = tuple(BYTE_TOKENS[value] for value in encoded)
        while len(tokens) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(tokens, tokens[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, selected_pair = min(ranked_pairs)
            tokens = _merge_pair(tokens, selected_pair, selected_pair[0] + selected_pair[1])
        return [self.token_to_id[token] for token in tokens]

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group())

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self.special_pattern is None:
            return list(self._encode_ordinary(text))

        token_ids: list[int] = []
        for piece in self.special_pattern.split(text):
            if not piece:
                continue
            special_id = self.special_to_id.get(piece)
            if special_id is not None:
                token_ids.append(special_id)
            else:
                token_ids.extend(self._encode_ordinary(piece))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if self.special_pattern is None:
            for text in iterable:
                yield from self.encode(text)
            return

        buffer = ""
        for text in iterable:
            buffer += text
            consumed = 0
            for match in self.special_pattern.finditer(buffer):
                yield from self._encode_ordinary(buffer[consumed : match.start()])
                yield self.special_to_id[match.group()]
                consumed = match.end()
            buffer = buffer[consumed:]
        yield from self._encode_ordinary(buffer)

    def decode(self, ids: list[int]) -> str:
        encoded = b"".join(self.vocab[token_id] for token_id in ids)
        return encoded.decode("utf-8", errors="replace")
