from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from pathlib import Path

import regex


GPT2_PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
BYTE_TOKENS = tuple(bytes((value,)) for value in range(256))


class _ReversePair(tuple[bytes, bytes]):
    """Order byte pairs in reverse lexicographic order for a min-heap."""

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, tuple):
            return NotImplemented
        return tuple.__gt__(self, other)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    if special_tokens:
        unique_specials = sorted(set(special_tokens), key=len, reverse=True)
        special_pattern = regex.compile("|".join(regex.escape(token) for token in unique_specials))
        ordinary_chunks = special_pattern.split(text)
    else:
        ordinary_chunks = [text]

    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in ordinary_chunks:
        for match in GPT2_PRETOKEN_PATTERN.finditer(chunk):
            encoded = match.group().encode("utf-8")
            counts[tuple(BYTE_TOKENS[value] for value in encoded)] += 1
    return counts


def _pretoken_counts_from_file(
    input_path: str | Path,
    special_tokens: list[str],
    chunk_size: int = 8 * 1024 * 1024,
) -> Counter[tuple[bytes, ...]]:
    if not special_tokens:
        return _pretoken_counts(Path(input_path).read_text(encoding="utf-8"), special_tokens)

    unique_specials = sorted(set(special_tokens), key=len, reverse=True)
    special_pattern = regex.compile("|".join(regex.escape(token) for token in unique_specials))
    counts: Counter[tuple[bytes, ...]] = Counter()

    def count_ordinary_chunk(chunk: str) -> None:
        for match in GPT2_PRETOKEN_PATTERN.finditer(chunk):
            encoded = match.group().encode("utf-8")
            counts[tuple(BYTE_TOKENS[value] for value in encoded)] += 1

    carry = ""
    with Path(input_path).open(encoding="utf-8") as input_file:
        while chunk := input_file.read(chunk_size):
            text = carry + chunk
            ordinary_start = 0
            for match in special_pattern.finditer(text):
                count_ordinary_chunk(text[ordinary_start : match.start()])
                ordinary_start = match.end()
            carry = text[ordinary_start:]

    count_ordinary_chunk(carry)
    return counts


def _merge_pair(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes], merged: bytes) -> tuple[bytes, ...]:
    output: list[bytes] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            output.append(merged)
            index += 2
        else:
            output.append(tokens[index])
            index += 1
    return tuple(output)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return its vocabulary and merge rules."""

    special_tokens = list(dict.fromkeys(special_tokens or []))
    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

    counted_words = _pretoken_counts_from_file(input_path, special_tokens)

    vocab: dict[int, bytes] = {index: token for index, token in enumerate(BYTE_TOKENS)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    words = list(counted_words)
    frequencies = [counted_words[word] for word in words]
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_id, (word, frequency) in enumerate(zip(words, frequencies, strict=True)):
        occurrences = Counter(zip(word, word[1:]))
        for pair, count in occurrences.items():
            pair_counts[pair] += count * frequency
            pair_to_words[pair].add(word_id)

    heap: list[tuple[int, _ReversePair, tuple[bytes, bytes]]] = [
        (-count, _ReversePair(pair), pair) for pair, count in pair_counts.items() if count > 0
    ]
    heapq.heapify(heap)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        selected_pair: tuple[bytes, bytes] | None = None
        while heap:
            negative_count, _, candidate = heapq.heappop(heap)
            if pair_counts.get(candidate, 0) == -negative_count and negative_count < 0:
                selected_pair = candidate
                break
        if selected_pair is None:
            break

        merged_token = selected_pair[0] + selected_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(selected_pair)

        affected_word_ids = tuple(pair_to_words.get(selected_pair, ()))
        touched_pairs: set[tuple[bytes, bytes]] = set()
        for word_id in affected_word_ids:
            old_word = words[word_id]
            frequency = frequencies[word_id]
            old_occurrences = Counter(zip(old_word, old_word[1:]))

            for pair, count in old_occurrences.items():
                pair_counts[pair] -= count * frequency
                pair_to_words[pair].discard(word_id)
                touched_pairs.add(pair)

            new_word = _merge_pair(old_word, selected_pair, merged_token)
            words[word_id] = new_word
            new_occurrences = Counter(zip(new_word, new_word[1:]))

            for pair, count in new_occurrences.items():
                pair_counts[pair] += count * frequency
                pair_to_words[pair].add(word_id)
                touched_pairs.add(pair)

        for pair in touched_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, (-count, _ReversePair(pair), pair))

    return vocab, merges
