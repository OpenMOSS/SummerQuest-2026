"""Byte-level BPE training utilities.

The implementation follows the Assignment 1 specification: GPT-2-style
pre-tokenization, all 256 bytes in the initial vocabulary, special tokens as
hard training boundaries, and deterministic lexicographic tie-breaking.
"""

from __future__ import annotations

import base64
import heapq
import json
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import regex


GPT2_PRETOKEN_PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

_BYTE_TOKENS = tuple(bytes((value,)) for value in range(256))
Pair = tuple[bytes, bytes]
Pretoken = tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class _ReversePair:
    """Order byte pairs from lexicographically greatest to smallest in a min-heap."""

    pair: Pair

    def __lt__(self, other: _ReversePair) -> bool:
        return self.pair > other.pair


def _unique_nonempty_special_tokens(special_tokens: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for token in special_tokens:
        if not isinstance(token, str):
            raise TypeError("special tokens must be strings")
        if not token:
            raise ValueError("special tokens must not be empty")
        if token not in seen:
            seen.add(token)
            result.append(token)
    return tuple(result)


def _special_splitter(special_tokens: tuple[str, ...]) -> Any:
    if not special_tokens:
        return None
    # Longest-first makes overlapping special tokens deterministic.
    ordered = sorted(special_tokens, key=lambda token: (-len(token), token))
    return regex.compile("|".join(regex.escape(token) for token in ordered))


def _count_pretokens(text: str, special_tokens: tuple[str, ...]) -> Counter[Pretoken]:
    """Count UTF-8 byte pre-tokens, excluding special-token spans."""

    counts: Counter[Pretoken] = Counter()
    splitter = _special_splitter(special_tokens)
    segments = (text,) if splitter is None else splitter.split(text)
    for segment in segments:
        for match in GPT2_PRETOKEN_PATTERN.finditer(segment):
            encoded = match.group().encode("utf-8")
            counts[tuple(_BYTE_TOKENS[value] for value in encoded)] += 1
    return counts


def _count_file_chunk(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[Pretoken]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        chunk = input_file.read(end - start)
    return _count_pretokens(chunk.decode("utf-8"), special_tokens)


def _find_chunk_boundaries(
    input_file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find safe chunk starts at occurrences of a document delimiter."""

    input_file.seek(0, os.SEEK_END)
    file_size = input_file.tell()
    input_file.seek(0)
    if file_size == 0 or desired_num_chunks <= 1:
        return [0, file_size]

    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [0]
    block_size = 64 * 1024
    overlap = max(0, len(split_special_token) - 1)

    for chunk_index in range(1, desired_num_chunks):
        position = chunk_index * chunk_size
        input_file.seek(position)
        carry = b""
        found: int | None = None
        while position < file_size:
            block = input_file.read(min(block_size, file_size - position))
            if not block:
                break
            searchable = carry + block
            match_index = searchable.find(split_special_token)
            if match_index >= 0:
                found = position - len(carry) + match_index
                break
            carry = searchable[-overlap:] if overlap else b""
            position += len(block)
        boundaries.append(file_size if found is None else found)

    boundaries.append(file_size)
    return sorted(set(boundaries))


def _pretoken_counts(
    input_path: str | os.PathLike[str],
    special_tokens: tuple[str, ...],
    num_workers: int,
) -> Counter[Pretoken]:
    path = os.fspath(input_path)
    if num_workers <= 1 or not special_tokens:
        with open(path, encoding="utf-8") as input_file:
            return _count_pretokens(input_file.read(), special_tokens)

    # Chunking is only safe at a known hard boundary.  Use the longest encoded
    # special token so the delimiter cannot occur strictly inside another
    # configured special token.  ``max`` keeps the user's order for equal
    # lengths, making the choice deterministic.
    encoded_specials = [token.encode("utf-8") for token in special_tokens]
    delimiter = encoded_specials[0]
    for candidate in encoded_specials[1:]:
        if len(candidate) > len(delimiter):
            delimiter = candidate
    with open(path, "rb") as input_file:
        boundaries = _find_chunk_boundaries(input_file, num_workers, delimiter)
    jobs = [
        (path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        if start < end
    ]
    if len(jobs) <= 1:
        return _count_file_chunk(jobs[0]) if jobs else Counter()

    counts: Counter[Pretoken] = Counter()
    context = mp.get_context("spawn")
    with context.Pool(processes=min(num_workers, len(jobs))) as pool:
        for partial_counts in pool.imap_unordered(_count_file_chunk, jobs):
            counts.update(partial_counts)
    return counts


def _adjacent_pairs(sequence: Pretoken) -> Iterable[Pair]:
    return zip(sequence, sequence[1:])


def _merge_pair(sequence: Pretoken, pair: Pair, merged_token: bytes) -> Pretoken:
    merged: list[bytes] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
            merged.append(merged_token)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | tuple[str, ...] | None = None,
    *,
    num_workers: int | None = 1,
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a deterministic byte-level BPE tokenizer.

    ``vocab_size`` includes the 256 byte tokens, special tokens, and tokens
    created by merges. ``num_processes`` is accepted as an alias for
    ``num_workers`` for convenient use from experiment scripts.
    """

    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    specials = _unique_nonempty_special_tokens(special_tokens or ())
    if num_processes is not None:
        if num_workers not in (None, 1) and num_workers != num_processes:
            raise ValueError("num_workers and num_processes disagree")
        num_workers = num_processes
    if num_workers is None:
        file_size = os.path.getsize(input_path)
        num_workers = min(16, os.cpu_count() or 1) if specials and file_size >= 10_000_000 else 1
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer or None")

    vocab: dict[int, bytes] = {token_id: _BYTE_TOKENS[token_id] for token_id in range(256)}
    existing_values = set(vocab.values())
    for special in specials:
        encoded = special.encode("utf-8")
        if encoded not in existing_values:
            vocab[len(vocab)] = encoded
            existing_values.add(encoded)
    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size={vocab_size} is too small for the {len(vocab)} initial byte and special tokens")

    pretoken_frequencies = _pretoken_counts(input_path, specials, num_workers)
    sequences = list(pretoken_frequencies)
    frequencies = [pretoken_frequencies[sequence] for sequence in sequences]

    pair_counts: dict[Pair, int] = defaultdict(int)
    pair_to_sequences: dict[Pair, set[int]] = defaultdict(set)
    for sequence_id, (sequence, frequency) in enumerate(zip(sequences, frequencies, strict=True)):
        for pair in _adjacent_pairs(sequence):
            pair_counts[pair] += frequency
            pair_to_sequences[pair].add(sequence_id)

    heap: list[tuple[int, _ReversePair, int, Pair]] = []
    serial = 0
    for pair, count in pair_counts.items():
        heap.append((-count, _ReversePair(pair), serial, pair))
        serial += 1
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(vocab) < vocab_size and heap:
        while heap:
            negative_count, _, _, best_pair = heapq.heappop(heap)
            count = -negative_count
            if count > 0 and pair_counts.get(best_pair, 0) == count:
                break
        else:
            break

        affected_sequences = tuple(pair_to_sequences.get(best_pair, ()))
        if not affected_sequences:
            pair_counts.pop(best_pair, None)
            continue

        merged_token = best_pair[0] + best_pair[1]
        changed_pairs: set[Pair] = set()
        for sequence_id in affected_sequences:
            old_sequence = sequences[sequence_id]
            frequency = frequencies[sequence_id]
            old_pairs = tuple(_adjacent_pairs(old_sequence))
            for pair in old_pairs:
                new_count = pair_counts[pair] - frequency
                if new_count:
                    pair_counts[pair] = new_count
                else:
                    pair_counts.pop(pair, None)
                changed_pairs.add(pair)
            for pair in set(old_pairs):
                sequence_ids = pair_to_sequences[pair]
                sequence_ids.discard(sequence_id)
                if not sequence_ids:
                    pair_to_sequences.pop(pair, None)

            new_sequence = _merge_pair(old_sequence, best_pair, merged_token)
            sequences[sequence_id] = new_sequence
            new_pairs = tuple(_adjacent_pairs(new_sequence))
            for pair in new_pairs:
                pair_counts[pair] += frequency
                changed_pairs.add(pair)
            for pair in set(new_pairs):
                pair_to_sequences[pair].add(sequence_id)

        vocab[len(vocab)] = merged_token
        merges.append(best_pair)
        for pair in changed_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, (-count, _ReversePair(pair), serial, pair))
                serial += 1

    return vocab, merges


def save_tokenizer_files(
    vocab: dict[int, bytes],
    merges: list[Pair],
    vocab_path: str | os.PathLike[str],
    merges_path: str | os.PathLike[str],
) -> None:
    """Serialize a tokenizer without relying on a lossy bytes-to-text mapping."""

    vocab_payload = {
        "format": "cs336-byte-bpe-v1",
        "vocab": [
            {"id": token_id, "bytes_base64": base64.b64encode(token_bytes).decode("ascii")}
            for token_id, token_bytes in sorted(vocab.items())
        ],
    }
    merges_payload = {
        "format": "cs336-byte-bpe-v1",
        "merges": [
            [base64.b64encode(left).decode("ascii"), base64.b64encode(right).decode("ascii")] for left, right in merges
        ],
    }
    Path(vocab_path).write_text(json.dumps(vocab_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    Path(merges_path).write_text(json.dumps(merges_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
