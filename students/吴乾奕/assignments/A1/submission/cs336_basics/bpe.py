"""Efficient, deterministic byte-pair encoding (BPE) training."""

from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .tokenizer import iter_pretokens


class _PairPriority:
    """A max-priority heap item ordered by count, then byte lexicography."""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: _PairPriority) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _read_text_chunks(input_path: str | os.PathLike[str], chunk_size: int) -> Iterator[str]:
    with Path(input_path).open(encoding="utf-8", newline="") as input_file:
        while chunk := input_file.read(chunk_size):
            yield chunk


def _adjacent_pair_counts(tokens: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    counts: dict[tuple[bytes, bytes], int] = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair(
    tokens: tuple[bytes, ...],
    selected_pair: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == selected_pair[0] and tokens[index + 1] == selected_pair[1]:
            merged.append(merged_token)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary.

    Pair counts are indexed by the word types that contain them.  A merge only
    revisits affected word types, and a lazy priority queue avoids scanning all
    pairs to find the next maximum.  Ties are resolved by choosing the
    lexicographically greatest pair, as required by the assignment.
    """

    special_tokens = [] if special_tokens is None else list(special_tokens)
    chunk_size = int(kwargs.pop("chunk_size", 1 << 20))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    vocab: dict[int, bytes] = {byte_value: bytes((byte_value,)) for byte_value in range(256)}
    vocab_values = set(vocab.values())
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")
        if special_bytes not in vocab_values:
            vocab[len(vocab)] = special_bytes
            vocab_values.add(special_bytes)

    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size={vocab_size} is smaller than the {len(vocab)} required base and special tokens")

    pretoken_frequencies: Counter[bytes] = Counter()
    chunks = _read_text_chunks(input_path, chunk_size)
    for piece, is_special in iter_pretokens(chunks, special_tokens):
        if not is_special:
            pretoken_frequencies[piece.encode("utf-8")] += 1

    single_byte_tokens = tuple(vocab[byte_value] for byte_value in range(256))
    words: list[tuple[bytes, ...]] = []
    word_frequencies: list[int] = []
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for raw_word, frequency in pretoken_frequencies.items():
        word = tuple(single_byte_tokens[byte_value] for byte_value in raw_word)
        word_id = len(words)
        words.append(word)
        word_frequencies.append(frequency)
        for pair, occurrences in _adjacent_pair_counts(word).items():
            pair_counts[pair] = pair_counts.get(pair, 0) + frequency * occurrences
            pair_to_words[pair].add(word_id)

    pair_heap = [_PairPriority(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        selected_pair: tuple[bytes, bytes] | None = None
        while pair_heap:
            candidate = heapq.heappop(pair_heap)
            if pair_counts.get(candidate.pair) == candidate.count:
                selected_pair = candidate.pair
                break
        if selected_pair is None:
            break

        merged_token = selected_pair[0] + selected_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(selected_pair)

        affected_word_ids = tuple(pair_to_words.get(selected_pair, ()))
        count_deltas: dict[tuple[bytes, bytes], int] = {}

        for word_id in affected_word_ids:
            old_word = words[word_id]
            old_pair_counts = _adjacent_pair_counts(old_word)
            new_word = _merge_pair(old_word, selected_pair, merged_token)
            new_pair_counts = _adjacent_pair_counts(new_word)
            words[word_id] = new_word
            frequency = word_frequencies[word_id]

            old_pairs = set(old_pair_counts)
            new_pairs = set(new_pair_counts)
            for pair in old_pairs - new_pairs:
                containing_words = pair_to_words.get(pair)
                if containing_words is not None:
                    containing_words.discard(word_id)
                    if not containing_words:
                        pair_to_words.pop(pair, None)
            for pair in new_pairs - old_pairs:
                pair_to_words[pair].add(word_id)

            for pair in old_pairs | new_pairs:
                delta = frequency * (new_pair_counts.get(pair, 0) - old_pair_counts.get(pair, 0))
                if delta:
                    count_deltas[pair] = count_deltas.get(pair, 0) + delta

        for pair, delta in count_deltas.items():
            updated_count = pair_counts.get(pair, 0) + delta
            if updated_count > 0:
                pair_counts[pair] = updated_count
                heapq.heappush(pair_heap, _PairPriority(updated_count, pair))
            else:
                pair_counts.pop(pair, None)

    return vocab, merges


# A descriptive alias is convenient for experiment scripts while ``train_bpe``
# remains the public adapter entry point.
run_train_bpe = train_bpe
