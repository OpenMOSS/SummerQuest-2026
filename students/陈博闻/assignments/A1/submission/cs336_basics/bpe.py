from __future__ import annotations

import os
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _gpt2_byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(codepoint): byte for byte, codepoint in zip(bs, cs)}


def _decode_gpt2_token(token: str, decoder: dict[str, int]) -> bytes:
    return bytes(decoder[char] for char in token)


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    if not special_tokens:
        yield text, False
        return

    pattern = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
    start = 0
    for match in pattern.finditer(text):
        if match.start() > start:
            yield text[start : match.start()], False
        yield match.group(0), True
        start = match.end()
    if start < len(text):
        yield text[start:], False


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for segment, is_special in _split_on_special_tokens(text, special_tokens):
        if is_special or not segment:
            continue
        for match in re.finditer(PAT, segment):
            counts[tuple(bytes([b]) for b in match.group(0).encode("utf-8"))] += 1
    return counts


def _pair_counts(word_counts: dict[tuple[bytes, ...], int]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, count in word_counts.items():
        for pair in zip(word, word[1:]):
            counts[pair] += count
    return counts


def _initial_pair_stats(
    word_counts: dict[tuple[bytes, ...], int],
) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, count in word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(word)
    return pair_counts, pair_to_words


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    if len(word) < 2:
        return word

    merged: list[bytes] = []
    i = 0
    left, right = pair
    new_token = left + right
    while i < len(word):
        if i + 1 < len(word) and word[i] == left and word[i + 1] == right:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    word_counts = dict(_pretoken_counts(text, special_tokens))
    pair_counts, pair_to_words = _initial_pair_stats(word_counts)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        if pair_counts[best_pair] <= 0:
            break
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_words = list(pair_to_words.pop(best_pair, set()))
        for word in affected_words:
            count = word_counts.pop(word, 0)
            if count == 0:
                continue

            new_word = _merge_word(word, best_pair)
            if new_word == word:
                word_counts[word] = word_counts.get(word, 0) + count
                continue

            for pair in zip(word, word[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word)

            word_counts[new_word] = word_counts.get(new_word, 0) + count
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += count
                pair_to_words[pair].add(new_word)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        existing = set(self.vocab.values())
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in existing:
                self.vocab[len(self.vocab)] = token_bytes
                existing.add(token_bytes)

        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            raw_vocab = json.load(f)

        decoder = _gpt2_byte_decoder()
        if all(str(key).isdigit() for key in raw_vocab):
            vocab = {int(token_id): bytes.fromhex(token_bytes) for token_id, token_bytes in raw_vocab.items()}
        else:
            vocab = {int(token_id): _decode_gpt2_token(token, decoder) for token, token_id in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((_decode_gpt2_token(parts[0], decoder), _decode_gpt2_token(parts[1], decoder)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment, is_special in _split_on_special_tokens(text, self.special_tokens):
            if not segment:
                continue
            if is_special:
                ids.append(self.bytes_to_id[segment.encode("utf-8")])
                continue
            for match in re.finditer(PAT, segment):
                ids.extend(self._encode_pretoken(match.group(0).encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        keep_chars = max(8192, 2 * max((len(token) for token in self.special_tokens), default=0))
        for text in iterable:
            buffer += text
            if len(buffer) <= 2 * keep_chars:
                continue

            cutoff = len(buffer) - keep_chars
            split_at = 0
            for match in re.finditer(PAT, buffer):
                if match.end() <= cutoff:
                    split_at = match.end()
                else:
                    break

            if split_at:
                yield from self.encode(buffer[:split_at])
                buffer = buffer[split_at:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def _encode_pretoken(self, token: bytes) -> tuple[int, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        parts = tuple(bytes([b]) for b in token)
        while len(parts) > 1:
            best_index = -1
            best_rank = len(self.merge_ranks)
            for i, pair in enumerate(zip(parts, parts[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_index = i
            if best_index == -1:
                break
            parts = parts[:best_index] + (parts[best_index] + parts[best_index + 1],) + parts[best_index + 2 :]

        ids = tuple(self.bytes_to_id[part] for part in parts)
        self._cache[token] = ids
        return ids
