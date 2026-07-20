"""Training utilities for a byte-level byte-pair encoding tokenizer."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from multiprocessing import get_context
from pathlib import Path

import regex

from cs336_basics.pretokenization_example import find_chunk_boundaries


PAT = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
_BYTE_TOKENS = tuple(bytes((value,)) for value in range(256))
_SPLIT_SPECIAL_TOKEN = b"<|endoftext|>"
_TARGET_CHUNK_BYTES = 64 * 1024 * 1024


def _unique_in_order(values: Iterable[str]) -> list[str]:
    """Return non-empty strings without duplicates, preserving input order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            raise ValueError("Special tokens must not be empty strings")
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[bytes]:
    """Pre-tokenize text while treating every special token as a hard boundary."""
    counts: Counter[bytes] = Counter()

    def add_segment(segment: str) -> None:
        for match in PAT.finditer(segment):
            counts[match.group().encode("utf-8")] += 1

    if not special_tokens:
        add_segment(text)
        return counts

    # Regex alternations are leftmost-first. Sorting by length ensures that an
    # overlapping longer special token wins over its shorter prefix.
    alternatives = sorted(special_tokens, key=len, reverse=True)
    special_pattern = regex.compile("|".join(regex.escape(token) for token in alternatives))

    segment_start = 0
    for match in special_pattern.finditer(text):
        add_segment(text[segment_start : match.start()])
        segment_start = match.end()
    add_segment(text[segment_start:])

    return counts


def _count_pretokens_in_file_range(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    """Worker entry point: count pre-tokens for one byte range."""
    input_path, start, end, special_tokens = args

    if end <= start:
        return Counter()

    with open(input_path, "rb") as file:
        file.seek(start)
        data = file.read(end - start)

    text = data.decode("utf-8")
    return _count_pretokens(text, list(special_tokens))


def _find_pretokenization_boundaries(input_path: Path, workers: int, special_tokens: list[str]) -> list[int]:
    """Find safe byte boundaries for independent pre-token counting."""
    file_size = input_path.stat().st_size
    if file_size == 0:
        return [0]

    # Exact parallel pre-tokenization needs a hard boundary. For OWT this is
    # <|endoftext|>. If that token is not configured as special, fall back to a
    # single exact chunk rather than changing tokenization semantics.
    if _SPLIT_SPECIAL_TOKEN.decode("utf-8") not in special_tokens:
        return [0, file_size]

    desired_num_chunks = max(workers, math.ceil(file_size / _TARGET_CHUNK_BYTES), 1)

    with input_path.open("rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            desired_num_chunks,
            _SPLIT_SPECIAL_TOKEN,
        )

    if not boundaries:
        return [0, file_size]

    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    if boundaries[-1] != file_size:
        boundaries.append(file_size)

    return boundaries


def _count_pretokens_from_file(input_path: str | Path, special_tokens: list[str], workers: int) -> Counter[bytes]:
    """Count pre-tokens from disk without reading the full corpus into memory."""
    if workers <= 0:
        raise ValueError("workers must be greater than 0")

    path = Path(input_path)
    boundaries = _find_pretokenization_boundaries(path, workers, special_tokens)
    tasks = [
        (str(path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries, boundaries[1:], strict=False)
        if start < end
    ]

    total: Counter[bytes] = Counter()
    if not tasks:
        return total

    if workers == 1 or len(tasks) == 1:
        for task in tasks:
            total.update(_count_pretokens_in_file_range(task))
        return total

    processes = min(workers, len(tasks))
    chunksize = max(1, len(tasks) // (processes * 4))

    with get_context().Pool(processes=processes) as pool:
        for partial_counts in pool.imap_unordered(_count_pretokens_in_file_range, tasks, chunksize=chunksize):
            total.update(partial_counts)

    return total


def _pair_multiplicities(tokens: list[bytes]) -> dict[tuple[bytes, bytes], int]:
    """Count adjacent pairs in one pre-token representation."""
    result: dict[tuple[bytes, bytes], int] = {}
    for left, right in zip(tokens, tokens[1:], strict=False):
        pair = (left, right)
        result[pair] = result.get(pair, 0) + 1
    return result


def _merge_pair(tokens: list[bytes], pair: tuple[bytes, bytes], merged: bytes) -> list[bytes]:
    """Merge all non-overlapping occurrences of ``pair`` from left to right."""
    left, right = pair
    result: list[bytes] = []
    index = 0

    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
            result.append(merged)
            index += 2
        else:
            result.append(tokens[index])
            index += 1

    return result


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    workers: int = 1,
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a deterministic byte-level BPE vocabulary.

    The initial vocabulary contains the supplied special tokens, in caller
    order, followed by all 256 one-byte tokens. During training, special-token
    occurrences split the corpus into independent regions and do not contribute
    to pair counts. Frequency ties are resolved by choosing the
    lexicographically greatest byte pair.

    ``workers=1`` counts pre-tokens sequentially. ``workers>1`` parallelizes
    pre-token counting across chunks split on <|endoftext|> hard boundaries.
    """
    if workers <= 0:
        raise ValueError("workers must be greater than 0")

    specials = _unique_in_order(special_tokens or [])

    vocab: dict[int, bytes] = {}
    token_to_id: dict[bytes, int] = {}

    def add_vocab_token(token: bytes) -> None:
        if token not in token_to_id:
            token_id = len(vocab)
            vocab[token_id] = token
            token_to_id[token] = token_id

    for special_token in specials:
        add_vocab_token(special_token.encode("utf-8"))
    for byte_token in _BYTE_TOKENS:
        add_vocab_token(byte_token)

    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size={vocab_size} is too small for the {len(vocab)} initial tokens")

    pretoken_counts = _count_pretokens_from_file(input_path, specials, workers)

    # Each distinct pre-token is represented once, with its corpus frequency
    # stored separately. Pair counts and the reverse pair-to-word index are then
    # updated only for words affected by a merge.
    words: list[list[bytes]] = []
    frequencies: list[int] = []
    for pretoken, frequency in pretoken_counts.items():
        words.append([_BYTE_TOKENS[value] for value in pretoken])
        frequencies.append(frequency)

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_id, tokens in enumerate(words):
        frequency = frequencies[word_id]
        for pair, multiplicity in _pair_multiplicities(tokens).items():
            pair_counts[pair] += frequency * multiplicity
            pair_to_words[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and pair_counts:
        # Pair comparison is part of the key so equal-frequency pairs choose the
        # lexicographically greatest tuple, as required by the assignment.
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merged_token = best_pair[0] + best_pair[1]
        affected_words = tuple(pair_to_words.get(best_pair, ()))

        merges.append(best_pair)

        for word_id in affected_words:
            old_tokens = words[word_id]
            frequency = frequencies[word_id]

            old_pairs = _pair_multiplicities(old_tokens)
            for pair, multiplicity in old_pairs.items():
                updated_count = pair_counts[pair] - frequency * multiplicity
                if updated_count:
                    pair_counts[pair] = updated_count
                else:
                    pair_counts.pop(pair, None)

                indexed_words = pair_to_words[pair]
                indexed_words.discard(word_id)
                if not indexed_words:
                    pair_to_words.pop(pair, None)

            new_tokens = _merge_pair(old_tokens, best_pair, merged_token)
            words[word_id] = new_tokens

            for pair, multiplicity in _pair_multiplicities(new_tokens).items():
                pair_counts[pair] += frequency * multiplicity
                pair_to_words[pair].add(word_id)

        add_vocab_token(merged_token)

    return vocab, merges


__all__ = ["PAT", "train_bpe"]