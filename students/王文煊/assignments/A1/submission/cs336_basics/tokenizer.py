"""Byte-level BPE: training (``train_bpe``) and a ``Tokenizer`` for encode/decode."""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from typing import BinaryIO

import regex as re


class _PairEntry:
    """Heap entry ordered by count desc, ties broken by greater byte pair.

    Byte tie-break matches the reference: among equal-count pairs the
    lexicographically greater ``(bytes, bytes)`` pair is merged first.
    """

    __slots__ = ("count", "a_bytes", "b_bytes", "pair")

    def __init__(self, count: int, a_bytes: bytes, b_bytes: bytes, pair: tuple[int, int]):
        self.count = count
        self.a_bytes = a_bytes
        self.b_bytes = b_bytes
        self.pair = pair

    def __lt__(self, other: "_PairEntry") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return (self.a_bytes, self.b_bytes) > (other.a_bytes, other.b_bytes)


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """Return chunk boundaries aligned to occurrences of ``split_special_token``.

    Adapted from the assignment's ``pretokenization_example`` so that chunks can be
    counted independently without ever splitting inside a document.
    """
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


# GPT-2 pre-tokenization pattern.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(PAT)


def _split_on_special(text: str, special_tokens: list[str]) -> list[str]:
    """Split ``text`` keeping special tokens as standalone segments.

    Longer special tokens take precedence so that e.g. ``<|eot|><|eot|>`` matches
    before ``<|eot|>`` when both are registered.
    """
    if not special_tokens:
        return [text]
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(tok) for tok in ordered) + ")"
    return re.split(pattern, text)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    """Count pre-tokens (as tuples of byte values) within one text chunk.

    Special tokens act as hard boundaries and are not counted.
    """
    counts: Counter[tuple[int, ...]] = Counter()
    special_set = set(special_tokens)
    for segment in _split_on_special(text, special_tokens):
        if segment in special_set or segment == "":
            continue
        for match in _COMPILED_PAT.finditer(segment):
            counts[tuple(match.group().encode("utf-8"))] += 1
    return counts


def _worker(args) -> Counter[tuple[int, ...]]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return _pretoken_counts(chunk, special_tokens)


def _parallel_pretoken_counts(
    input_path: str | os.PathLike, special_tokens: list[str], num_processes: int
) -> Counter[tuple[int, ...]]:
    split_token = (special_tokens[0] if special_tokens else "<|endoftext|>").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    jobs = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    total: Counter[tuple[int, ...]] = Counter()
    if len(jobs) <= 1 or num_processes <= 1:
        for job in jobs:
            total.update(_worker(job))
    else:
        with Pool(min(num_processes, len(jobs))) as pool:
            for partial in pool.imap_unordered(_worker, jobs):
                total.update(partial)
    return total


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns ``(vocab, merges)`` where ``vocab`` maps id -> bytes and ``merges`` is
    the ordered list of merged byte pairs.
    """
    # 1. Initial vocabulary: 256 raw bytes, then special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # 2. Pre-tokenize into (word -> frequency), words are tuples of token ids.
    raw_counts = _parallel_pretoken_counts(input_path, special_tokens, num_processes)
    words: list[list[int]] = []
    freqs: list[int] = []
    for word_bytes, freq in raw_counts.items():
        words.append(list(word_bytes))
        freqs.append(freq)

    # 3. Pair statistics with an index from pair -> set of word indices,
    #    plus a lazy-deletion max-heap so each merge finds the best pair in
    #    ~O(log H) instead of scanning every distinct pair.
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[int]] = {}

    for wi in range(len(words)):
        symbols = words[wi]
        f = freqs[wi]
        for a, b in zip(symbols, symbols[1:]):
            pair = (a, b)
            pair_counts[pair] += f
            pair_to_words.setdefault(pair, set()).add(wi)

    heap: list[_PairEntry] = [
        _PairEntry(c, vocab[p[0]], vocab[p[1]], p) for p, c in pair_counts.items() if c > 0
    ]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        # Pop stale entries until we find one matching the current count.
        best_pair = None
        while heap:
            entry = heapq.heappop(heap)
            if entry.count > 0 and pair_counts.get(entry.pair, 0) == entry.count:
                best_pair = entry.pair
                break
        if best_pair is None:
            break

        a, b = best_pair
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        changed: set[tuple[int, int]] = set()
        affected = list(pair_to_words.get(best_pair, ()))
        for wi in affected:
            symbols = words[wi]
            f = freqs[wi]
            # Remove this word's old pair contributions.
            for x, y in zip(symbols, symbols[1:]):
                pair_counts[(x, y)] -= f
                changed.add((x, y))
                s = pair_to_words.get((x, y))
                if s is not None:
                    s.discard(wi)
            # Merge every occurrence of (a, b) within this word.
            merged: list[int] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            words[wi] = merged
            # Add the word's new pair contributions.
            for x, y in zip(merged, merged[1:]):
                pair_counts[(x, y)] += f
                changed.add((x, y))
                pair_to_words.setdefault((x, y), set()).add(wi)

        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)
        changed.discard(best_pair)

        # Push refreshed heap entries for every pair whose count changed.
        for p in changed:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, _PairEntry(c, vocab[p[0]], vocab[p[1]], p))

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer parameterized by a vocab, merges and special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        self.token_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        # Ensure special tokens have ids (append any that are missing).
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = b
                self.token_to_id[b] = new_id
        self.special_ids = {tok: self.token_to_id[tok.encode("utf-8")] for tok in self.special_tokens}

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            raw = json.load(f)
        vocab = {int(k): bytes(v) for k, v in raw.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as f:
            for line in f:
                parts = line.rstrip(b"\n").split(b" ")
                if len(parts) == 2:
                    merges.append((bytes(json.loads(parts[0])), bytes(json.loads(parts[1]))))
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, token_bytes: bytes) -> list[int]:
        """BPE-merge a single pre-token's bytes into a list of ids."""
        parts = [bytes([b]) for b in token_bytes]
        while len(parts) > 1:
            best_rank = None
            best_i = None
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return [self.token_to_id[p] for p in parts]

    def _encode_chunk(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _COMPILED_PAT.finditer(text):
            ids.extend(self._apply_merges(match.group().encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment in _split_on_special(text, self.special_tokens):
            if segment == "":
                continue
            if segment in self.special_ids:
                ids.append(self.special_ids[segment])
            else:
                ids.extend(self._encode_chunk(segment))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Stream-encode an iterable of strings (e.g. a file), bounded memory."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab.get(i, b"") for i in ids)
        return data.decode("utf-8", errors="replace")
