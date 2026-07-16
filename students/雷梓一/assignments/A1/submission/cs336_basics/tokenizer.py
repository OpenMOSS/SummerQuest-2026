from __future__ import annotations

import json
import heapq
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex

from .pretokenization_example import find_chunk_boundaries


GPT2_PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


@dataclass(frozen=True)
class _DescendingPair:
    """Reverse a pair's lexicographic order for use in Python's min-heap."""

    pair: tuple[bytes, bytes]

    def __lt__(self, other: _DescendingPair) -> bool:
        return self.pair > other.pair


def _special_pattern(special_tokens: list[str]) -> Any | None:
    if not special_tokens:
        return None
    alternatives = "|".join(regex.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True))
    return regex.compile(f"({alternatives})")


def _ordinary_sections(text: str, special_tokens: list[str]) -> Iterator[str]:
    pattern = _special_pattern(special_tokens)
    if pattern is None:
        yield text
        return
    for section in pattern.split(text):
        if section and section not in special_tokens:
            yield section


def _count_text_pretokens(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()

    for section in _ordinary_sections(text, special_tokens):
        for match in GPT2_PRETOKEN_PATTERN.finditer(section):
            counts[tuple(bytes([byte]) for byte in match.group().encode("utf-8"))] += 1
    return counts


def _count_file_chunk(args: tuple[str, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors="ignore")
    return _count_text_pretokens(text, special_tokens)


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
) -> Counter[tuple[bytes, ...]]:
    path = Path(input_path)
    num_processes = max(1, num_processes)

    file_size = path.stat().st_size
    if file_size <= 64 * 1024 * 1024 or not special_tokens or num_processes == 1:
        return _count_text_pretokens(path.read_text(encoding="utf-8"), special_tokens)

    split_token = special_tokens[0].encode("utf-8")
    with path.open("rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes * 4, split_token)
    tasks = [
        (str(path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]
    counts: Counter[tuple[bytes, ...]] = Counter()
    with mp.get_context("fork").Pool(processes=min(num_processes, len(tasks))) as pool:
        for chunk_counts in pool.imap(_count_file_chunk, tasks):
            counts.update(chunk_counts)
    return counts


def _merge_pair(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = list(dict.fromkeys(special_tokens or []))
    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

    vocabulary: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    next_id = 256
    existing_values = set(vocabulary.values())
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in existing_values:
            vocabulary[next_id] = token_bytes
            existing_values.add(token_bytes)
            next_id += 1

    if num_processes is None:
        num_processes = min(16, os.cpu_count() or 1)
    pretoken_counts = _count_pretokens(input_path, special_tokens, num_processes)
    words = list(pretoken_counts.keys())
    frequencies = [pretoken_counts[word] for word in words]
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_index, (word, frequency) in enumerate(zip(words, frequencies)):
        seen_pairs: set[tuple[bytes, bytes]] = set()
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency
            seen_pairs.add(pair)
        for pair in seen_pairs:
            pair_to_words[pair].add(word_index)

    pair_heap = [(-count, _DescendingPair(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocabulary) < vocab_size and pair_heap:
        while pair_heap:
            negative_count, _descending_pair, candidate = heapq.heappop(pair_heap)
            if pair_counts.get(candidate, 0) == -negative_count:
                best_pair = candidate
                break
        else:
            break
        affected_words = tuple(pair_to_words.get(best_pair, ()))
        if not affected_words:
            del pair_counts[best_pair]
            continue

        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocabulary[len(vocabulary)] = merged_token

        touched_pairs: set[tuple[bytes, bytes]] = {best_pair}
        for word_index in affected_words:
            old_word = words[word_index]
            frequency = frequencies[word_index]
            old_pairs = list(zip(old_word, old_word[1:]))
            touched_pairs.update(old_pairs)
            for pair in old_pairs:
                pair_counts[pair] -= frequency
            for pair in set(old_pairs):
                pair_to_words[pair].discard(word_index)

            new_word = _merge_pair(old_word, best_pair)
            words[word_index] = new_word
            new_pairs = list(zip(new_word, new_word[1:]))
            touched_pairs.update(new_pairs)
            for pair in new_pairs:
                pair_counts[pair] += frequency
            for pair in set(new_pairs):
                pair_to_words[pair].add(word_index)

        for pair in touched_pairs:
            if pair_counts.get(pair, 0) <= 0:
                pair_counts.pop(pair, None)
                pair_to_words.pop(pair, None)
            else:
                heapq.heappush(pair_heap, (-pair_counts[pair], _DescendingPair(pair), pair))

    return vocabulary, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        values_to_ids = {token: token_id for token_id, token in self.vocab.items()}
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in values_to_ids:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = special_bytes
                values_to_ids[special_bytes] = new_id
        self.bytes_to_id = values_to_ids
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_pattern = _special_pattern(self.special_tokens)
        self._special_ids = {
            token: self.bytes_to_id[token.encode("utf-8")] for token in self.special_tokens
        }

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_path, encoding="utf-8") as file:
            serialized_vocab = json.load(file)
        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in serialized_vocab.items()}
        with open(merges_path, encoding="utf-8") as file:
            serialized_merges = json.load(file)
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in serialized_merges]
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_path: str | os.PathLike, merges_path: str | os.PathLike) -> None:
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump({str(token_id): token.hex() for token_id, token in self.vocab.items()}, file, indent=2)
        with open(merges_path, "w", encoding="utf-8") as file:
            json.dump([[left.hex(), right.hex()] for left, right in self.merges], file, indent=2)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
        while len(tokens) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(tokens, tokens[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            tokens = _merge_pair(tokens, best_pair)
        return [self.bytes_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        sections = self._special_pattern.split(text) if self._special_pattern is not None else [text]
        for section in sections:
            if not section:
                continue
            special_id = self._special_ids.get(section)
            if special_id is not None:
                encoded.append(special_id)
                continue
            for match in GPT2_PRETOKEN_PATTERN.finditer(section):
                encoded.extend(self._encode_pretoken(match.group()))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        try:
            data = b"".join(self.vocab[int(token_id)] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return data.decode("utf-8", errors="replace")
