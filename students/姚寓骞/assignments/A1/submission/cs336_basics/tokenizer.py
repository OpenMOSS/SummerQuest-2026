from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from heapq import heapify, heappop, heappush
from pathlib import Path

import regex


GPT2_PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


class _MaxPairEntry:
    """A max-heap entry ordered by pair frequency, then lexicographically."""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: "_MaxPairEntry") -> bool:
        return (self.count, self.pair) > (other.count, other.pair)


def _count_pretokens(text: str, special_tokens: tuple[str, ...]) -> Counter[bytes]:
    """Count encoded pre-tokens without materializing every byte-token tuple."""
    frequencies: Counter[bytes] = Counter()
    if not special_tokens:
        for match in GPT2_PATTERN.finditer(text):
            frequencies[match.group().encode("utf-8")] += 1
        return frequencies

    splitter = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
    start = 0
    for special_match in splitter.finditer(text):
        for match in GPT2_PATTERN.finditer(text, start, special_match.start()):
            frequencies[match.group().encode("utf-8")] += 1
        start = special_match.end()
    for match in GPT2_PATTERN.finditer(text, start):
        frequencies[match.group().encode("utf-8")] += 1
    return frequencies


def _count_pretokens_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: tuple[str, ...],
) -> Counter[bytes]:
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")
    return _count_pretokens(text, special_tokens)


def _find_chunk_boundaries(input_path: str | Path, desired_num_chunks: int, split_token: bytes) -> list[int]:
    """Find UTF-8-safe chunk boundaries at occurrences of a special token."""
    with open(input_path, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        if file_size == 0:
            return [0]

        chunk_size = max(1, file_size // desired_num_chunks)
        boundaries = [i * chunk_size for i in range(desired_num_chunks)] + [file_size]
        read_size = 64 * 1024
        overlap = max(0, len(split_token) - 1)

        for index in range(1, len(boundaries) - 1):
            position = boundaries[index]
            file.seek(position)
            carry = b""
            found_boundary = False
            while position < file_size:
                data = file.read(read_size)
                if not data:
                    break
                window = carry + data
                found_at = window.find(split_token)
                if found_at >= 0:
                    boundaries[index] = position - len(carry) + found_at
                    found_boundary = True
                    break
                carry = window[-overlap:] if overlap else b""
                position += len(data)
            if not found_boundary:
                boundaries[index] = file_size

    return sorted(set(boundaries))


def _pretoken_frequencies(
    input_path: str | Path,
    special_tokens: list[str],
    num_processes: int,
) -> Counter[bytes]:
    specials = tuple(special_tokens)
    if num_processes <= 1 or not special_tokens:
        return _count_pretokens(Path(input_path).read_text(encoding="utf-8"), specials)

    # More chunks than workers bounds per-process text memory and balances uneven documents.
    split_token = max(special_tokens, key=len).encode("utf-8")
    boundaries = _find_chunk_boundaries(input_path, num_processes * 4, split_token)
    ranges = [(start, end) for start, end in zip(boundaries, boundaries[1:]) if start < end]
    if len(ranges) <= 1:
        return _count_pretokens(Path(input_path).read_text(encoding="utf-8"), specials)

    frequencies: Counter[bytes] = Counter()
    with ProcessPoolExecutor(max_workers=min(num_processes, len(ranges))) as executor:
        futures = (
            executor.submit(_count_pretokens_chunk, str(input_path), start, end, specials) for start, end in ranges
        )
        for future in as_completed(futures):
            frequencies.update(future.result())
    return frequencies


def _merge_pair(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    result: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return tuple(result)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE with deterministic, lexicographic tie breaking."""
    special_bytes = [token.encode("utf-8") for token in special_tokens]
    if vocab_size < 256 + len(special_bytes):
        raise ValueError("vocab_size is too small for bytes and special tokens")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")

    pretoken_frequencies = _pretoken_frequencies(input_path, special_tokens, num_processes)
    frequencies = {
        tuple(map(BYTE_TOKENS.__getitem__, pretoken)): frequency
        for pretoken, frequency in pretoken_frequencies.items()
    }

    words = list(frequencies)
    word_freq = [frequencies[word] for word in words]
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    def word_pairs(word: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
        return Counter(zip(word, word[1:]))

    word_pair_counts = [word_pairs(word) for word in words]
    for word_id, pairs in enumerate(word_pair_counts):
        for pair, count in pairs.items():
            pair_counts[pair] += count * word_freq[word_id]
            pair_words[pair].add(word_id)

    pair_heap = [_MaxPairEntry(count, pair) for pair, count in pair_counts.items()]
    heapify(pair_heap)

    vocab: dict[int, bytes] = dict(enumerate(BYTE_TOKENS))
    for token in special_bytes:
        if token not in vocab.values():
            vocab[len(vocab)] = token
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and pair_counts:
        while pair_heap:
            entry = heappop(pair_heap)
            if pair_counts.get(entry.pair) == entry.count:
                break
        else:
            break
        best_pair = entry.pair
        affected = list(pair_words.get(best_pair, ()))
        if not affected or pair_counts[best_pair] <= 0:
            pair_counts.pop(best_pair, None)
            continue

        changed_pairs: set[tuple[bytes, bytes]] = set()
        for word_id in affected:
            old_word = words[word_id]
            old_pairs = word_pair_counts[word_id]
            for pair, count in old_pairs.items():
                pair_counts[pair] -= count * word_freq[word_id]
                pair_words[pair].discard(word_id)
                changed_pairs.add(pair)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            new_word = _merge_pair(old_word, best_pair)
            words[word_id] = new_word
            new_pairs = word_pairs(new_word)
            word_pair_counts[word_id] = new_pairs
            for pair, count in new_pairs.items():
                pair_counts[pair] += count * word_freq[word_id]
                pair_words[pair].add(word_id)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            count = pair_counts.get(pair)
            if count is not None:
                heappush(pair_heap, _MaxPairEntry(count, pair))
        if len(pair_heap) > max(1_000, 4 * len(pair_counts)):
            pair_heap = [_MaxPairEntry(count, pair) for pair, count in pair_counts.items()]
            heapify(pair_heap)

        merged = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged
        merges.append(best_pair)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.bytes_to_id = {value: key for key, value in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.special_to_id = {token: self.bytes_to_id[token.encode("utf-8")] for token in self.special_tokens}
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(token) for token in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )

    def _encode_piece(self, piece: str) -> list[int]:
        tokens = tuple(map(BYTE_TOKENS.__getitem__, piece.encode("utf-8")))
        while len(tokens) > 1:
            ranked = [(self.merge_ranks[pair], pair) for pair in zip(tokens, tokens[1:]) if pair in self.merge_ranks]
            if not ranked:
                break
            _, pair = min(ranked)
            tokens = _merge_pair(tokens, pair)
        return [self.bytes_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        parts = self._special_pattern.split(text) if self._special_pattern else [text]
        for part in parts:
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                for match in GPT2_PATTERN.finditer(part):
                    ids.extend(self._encode_piece(match.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        try:
            data = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token id: {exc.args[0]}") from exc
        return data.decode("utf-8", errors="replace")
