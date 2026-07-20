from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import json
import os

import regex


PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _merge_pair(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    result: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(pair[0] + pair[1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return tuple(result)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.bytes_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.special_to_id = {
            token: self.bytes_to_id[token.encode("utf-8")] for token in self.special_tokens
        }
        self._special_pattern = (
            regex.compile("(" + "|".join(regex.escape(token) for token in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as f:
            encoded_vocab = json.load(f)
        vocab = {int(key): bytes(value) for key, value in encoded_vocab.items()}
        with open(merges_filepath, encoding="utf-8") as f:
            encoded_merges = json.load(f)
        merges = [(bytes(left), bytes(right)) for left, right in encoded_merges]
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath, merges_filepath) -> None:
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump({key: list(value) for key, value in self.vocab.items()}, f)
        ordered_merges = sorted(self.merge_ranks, key=self.merge_ranks.get)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump([[list(left), list(right)] for left, right in ordered_merges], f)

    def _encode_bytes(self, data: bytes) -> list[int]:
        tokens = tuple(bytes([byte]) for byte in data)
        while len(tokens) > 1:
            candidates = {
                pair: self.merge_ranks[pair]
                for pair in zip(tokens, tokens[1:])
                if pair in self.merge_ranks
            }
            if not candidates:
                break
            pair = min(candidates, key=candidates.get)
            tokens = _merge_pair(tokens, pair)
        return [self.bytes_to_id[token] for token in tokens]

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in PATTERN.finditer(text):
            yield from self._encode_bytes(match.group().encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        if self._special_pattern is None:
            return list(self._encode_ordinary(text))
        result: list[int] = []
        for part in self._special_pattern.split(text):
            if not part:
                continue
            if part in self.special_to_id:
                result.append(self.special_to_id[part])
            else:
                result.extend(self._encode_ordinary(part))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small for byte tokens and special tokens")

    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        special_pattern = regex.compile("|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True)))
        ordinary_parts = special_pattern.split(text)
    else:
        ordinary_parts = [text]

    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
    for part in ordinary_parts:
        for match in PATTERN.finditer(part):
            pretoken_counts[tuple(bytes([byte]) for byte in match.group().encode("utf-8"))] += 1

    words = list(pretoken_counts)
    frequencies = [pretoken_counts[word] for word in words]
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    count_to_pairs: dict[int, set[tuple[bytes, bytes]]] = defaultdict(set)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    def change_count(pair: tuple[bytes, bytes], delta: int) -> None:
        old = pair_counts.get(pair, 0)
        if old:
            count_to_pairs[old].discard(pair)
            if not count_to_pairs[old]:
                del count_to_pairs[old]
        new = old + delta
        if new:
            pair_counts[pair] = new
            count_to_pairs[new].add(pair)
        else:
            pair_counts.pop(pair, None)

    for index, (word, frequency) in enumerate(zip(words, frequencies)):
        occurrences = Counter(zip(word, word[1:]))
        for pair, count in occurrences.items():
            change_count(pair, count * frequency)
            pair_to_words[pair].add(index)

    vocab = {index: bytes([index]) for index in range(256)}
    for token in special_tokens:
        encoded = token.encode("utf-8")
        if encoded not in vocab.values():
            vocab[len(vocab)] = encoded
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and count_to_pairs:
        max_count = max(count_to_pairs)
        pair = max(count_to_pairs[max_count])
        affected = list(pair_to_words.get(pair, ()))
        for index in affected:
            old_word = words[index]
            old_occurrences = Counter(zip(old_word, old_word[1:]))
            for old_pair, count in old_occurrences.items():
                change_count(old_pair, -count * frequencies[index])
                pair_to_words[old_pair].discard(index)

            new_word = _merge_pair(old_word, pair)
            words[index] = new_word
            new_occurrences = Counter(zip(new_word, new_word[1:]))
            for new_pair, count in new_occurrences.items():
                change_count(new_pair, count * frequencies[index])
                pair_to_words[new_pair].add(index)

        merged_token = pair[0] + pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(pair)

    return vocab, merges
