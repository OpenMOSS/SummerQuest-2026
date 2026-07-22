from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

try:
    import regex
except ImportError:  # pragma: no cover
    regex = None


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _compile_pattern():
    if regex is None:
        raise RuntimeError("The regex package is required for GPT-2 pre-tokenization.")
    return regex.compile(PATTERN)


def _split_on_specials(text: str, special_tokens: list[str]) -> list[tuple[str, bool]]:
    if not special_tokens:
        return [(text, False)]
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(token) for token in ordered))
    parts: list[tuple[str, bool]] = []
    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            parts.append((text[position : match.start()], False))
        parts.append((match.group(0), True))
        position = match.end()
    if position < len(text):
        parts.append((text[position:], False))
    return parts


def _pretokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[bytes]:
    pattern = _compile_pattern()
    for part, is_special in _split_on_specials(text, special_tokens or []):
        if is_special:
            continue
        for match in pattern.finditer(part):
            yield match.group(0).encode("utf-8")


def _word_to_symbols(word: bytes) -> tuple[bytes, ...]:
    return tuple(bytes([byte]) for byte in word)


def _count_pairs(word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, count in word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
    return pair_counts


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    index = 0
    while index < len(word):
        if index < len(word) - 1 and word[index] == pair[0] and word[index + 1] == pair[1]:
            merged.append(word[index] + word[index + 1])
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    for token in special_tokens:
        encoded = token.encode("utf-8")
        if encoded not in vocab.values():
            vocab[len(vocab)] = encoded

    with open(input_path, encoding="utf-8") as file:
        text = file.read()

    word_counts: Counter[tuple[int, ...]] = Counter(tuple(pretoken) for pretoken in _pretokenize(text, special_tokens))
    words = [list(word) for word in word_counts]
    counts = [word_counts[tuple(word)] for word in words]
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for word_idx, word in enumerate(words):
        count = counts[word_idx]
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(word_idx)
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], (vocab[pair[0]], vocab[pair[1]])))
        merged_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        new_token_id = len(vocab)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_token_id] = merged_bytes
        affected = list(pair_to_words.pop(best_pair, set()))
        pair_counts.pop(best_pair, None)
        for word_idx in affected:
            word = words[word_idx]
            count = counts[word_idx]
            for pair in zip(word, word[1:]):
                if pair in pair_counts:
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word_idx)
                    if not pair_to_words[pair]:
                        pair_to_words.pop(pair, None)
            merged: list[int] = []
            index = 0
            while index < len(word):
                if index < len(word) - 1 and word[index] == best_pair[0] and word[index + 1] == best_pair[1]:
                    merged.append(new_token_id)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            words[word_idx] = merged
            for pair in zip(merged, merged[1:]):
                pair_counts[pair] += count
                pair_to_words[pair].add(word_idx)
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.byte_to_id = {token: idx for idx, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_bytes_to_id = {
            token.encode("utf-8"): self.byte_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self.byte_to_id
        }
        self._encode_cache: dict[bytes, list[int]] = {}
        self._encode_cache_limit = 262_144

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        cached = self._encode_cache.get(pretoken)
        if cached is not None:
            return cached
        symbols = _word_to_symbols(pretoken)
        if len(symbols) == 0:
            return []
        while True:
            pairs = list(zip(symbols, symbols[1:]))
            ranked_pairs = [(self.merge_ranks[pair], pair) for pair in pairs if pair in self.merge_ranks]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs)
            symbols = _merge_word(symbols, best_pair)
        ids = [self.byte_to_id[symbol] for symbol in symbols]
        if len(self._encode_cache) >= self._encode_cache_limit:
            self._encode_cache.clear()
        self._encode_cache[pretoken] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        pattern = _compile_pattern()
        for part, is_special in _split_on_specials(text, self.special_tokens):
            if is_special:
                ids.append(self.byte_to_id[part.encode("utf-8")])
            else:
                for match in pattern.finditer(part):
                    ids.extend(self._encode_pretoken(match.group(0).encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int] | Iterable[int]) -> str:
        content = b"".join(self.vocab[idx] for idx in ids)
        return content.decode("utf-8", errors="replace")
