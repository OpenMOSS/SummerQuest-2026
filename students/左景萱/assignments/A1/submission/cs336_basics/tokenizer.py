"""Byte-level BPE training and tokenization utilities.

The implementation uses the GPT-2 pre-tokenization expression from the
assignment. BPE itself always operates on raw bytes; the reversible Unicode
mapping is only used when reading or writing conventional GPT-2 files.
"""

from __future__ import annotations

import json
import heapq
import multiprocessing as mp
import os
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

import regex


GPT2_PRETOKEN_PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
_BYTE_TOKENS = tuple(bytes([value]) for value in range(256))


class _PairPriority:
    """Heap entry whose smallest value is the assignment's best BPE pair."""

    __slots__ = ("frequency", "pair")

    def __init__(self, frequency: int, pair: tuple[bytes, bytes]) -> None:
        self.frequency = frequency
        self.pair = pair

    def __lt__(self, other: _PairPriority) -> bool:
        if self.frequency != other.frequency:
            return self.frequency > other.frequency
        return self.pair > other.pair


def _pair_heap(pair_counts: Counter[tuple[bytes, bytes]]) -> list[_PairPriority]:
    heap = [_PairPriority(frequency, pair) for pair, frequency in pair_counts.items()]
    heapq.heapify(heap)
    return heap


def _pop_best_pair(
    heap: list[_PairPriority],
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    """Pop the best current pair, discarding lazily invalidated entries."""

    while heap:
        candidate = heapq.heappop(heap)
        if pair_counts.get(candidate.pair) == candidate.frequency:
            return candidate.pair
    raise RuntimeError("pair heap exhausted while pair counts remain")


def _unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _special_pattern(special_tokens: Iterable[str]) -> regex.Pattern[str] | None:
    """Build a leftmost-longest pattern for literal special tokens."""

    tokens = sorted(set(special_tokens), key=lambda token: (-len(token), token))
    if not tokens:
        return None
    return regex.compile("|".join(regex.escape(token) for token in tokens))


def _bytes_to_unicode() -> dict[int, str]:
    """Return OpenAI's reversible byte-to-Unicode map used by GPT-2 files."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, codepoints), strict=True))


def _split_around_specials(text: str, pattern: regex.Pattern[str] | None) -> Iterator[str]:
    """Yield ordinary-text spans, omitting every special-token occurrence."""

    if pattern is None:
        if text:
            yield text
        return

    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            yield text[cursor : match.start()]
        cursor = match.end()
    if cursor < len(text):
        yield text[cursor:]


def _count_text_pretokens(text: str, special_tokens: tuple[str, ...]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    pattern = _special_pattern(special_tokens)
    for ordinary_text in _split_around_specials(text, pattern):
        counts.update(match.group(0).encode("utf-8") for match in GPT2_PRETOKEN_PATTERN.finditer(ordinary_text))
    return counts


def _count_file_range(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    path, start, end, special_tokens = args
    with open(path, "rb") as file:
        file.seek(start)
        contents = file.read(end - start).decode("utf-8", errors="ignore")
    return _count_text_pretokens(contents, special_tokens)


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_token: bytes,
) -> list[int]:
    """Find byte ranges that begin at a document delimiter (or newline)."""

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size == 0:
        return [0]

    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [index * chunk_size for index in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    for index in range(1, len(boundaries) - 1):
        position = boundaries[index]
        file.seek(position)
        carry = b""
        while position < file_size:
            block = file.read(min(64 * 1024, file_size - position))
            if not block:
                boundaries[index] = file_size
                break
            searchable = carry + block
            found = searchable.find(split_token)
            if found >= 0:
                boundaries[index] = position - len(carry) + found
                break
            keep = max(0, len(split_token) - 1)
            carry = searchable[-keep:] if keep else b""
            position += len(block)

    return sorted(set(boundaries))


def _count_corpus_pretokens(
    input_path: str | os.PathLike[str],
    special_tokens: tuple[str, ...],
    num_processes: int,
) -> Counter[bytes]:
    """Count pre-tokens without loading a large corpus in the parent process."""

    path = os.fspath(input_path)
    file_size = os.path.getsize(path)
    target_chunk_size = 64 * 1024 * 1024

    # Keeping small inputs intact is both faster and exactly preserves regex
    # behavior for whitespace runs that span line boundaries.
    if file_size <= target_chunk_size:
        return _count_file_range((path, 0, file_size, special_tokens))

    if num_processes > 1:
        # Assignment corpora are document-delimited by <|endoftext|>. Falling
        # back to a newline still keeps UTF-8 chunk boundaries valid.
        delimiter = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        with open(path, "rb") as file:
            boundaries = _find_chunk_boundaries(file, num_processes, delimiter)
        ranges = [
            (path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            if end > start
        ]
        if len(ranges) > 1:
            counts: Counter[bytes] = Counter()
            with mp.get_context("spawn").Pool(min(num_processes, len(ranges))) as pool:
                for partial_counts in pool.imap_unordered(_count_file_range, ranges):
                    counts.update(partial_counts)
            return counts

    # A removed special token is an exact pre-tokenization boundary. Process
    # large corpora in bounded chunks at those boundaries, even with one
    # worker, so no regex match is changed by the streaming implementation.
    if special_tokens:
        desired_chunks = max(2, (file_size + target_chunk_size - 1) // target_chunk_size)
        with open(path, "rb") as file:
            boundaries = _find_chunk_boundaries(file, desired_chunks, special_tokens[0].encode("utf-8"))
        counts: Counter[bytes] = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            counts.update(_count_file_range((path, start, end, special_tokens)))
        return counts

    # Line-wise processing bounds memory on multi-gigabyte corpora. Newlines
    # remain part of each line, so no corpus bytes are dropped.
    counts = Counter()
    pattern = _special_pattern(special_tokens)
    with open(path, encoding="utf-8") as file:
        for line in file:
            for ordinary_text in _split_around_specials(line, pattern):
                counts.update(match.group(0).encode("utf-8") for match in GPT2_PRETOKEN_PATTERN.finditer(ordinary_text))
    return counts


def _merge_pair(symbols: tuple[bytes, ...], pair: tuple[bytes, bytes], merged: bytes) -> tuple[bytes, ...]:
    """Merge every non-overlapping occurrence of ``pair`` from left to right."""

    output: list[bytes] = []
    index = 0
    first, second = pair
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index] == first and symbols[index + 1] == second:
            output.append(merged)
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return tuple(output)


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    *,
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary.

    Pair frequencies count repeated occurrences within a pre-token. Frequency
    ties choose the lexicographically greatest byte pair, per the assignment.
    """

    specials = tuple(_unique_in_order(special_tokens or []))
    initial_vocab_size = len(specials) + 256
    if vocab_size < initial_vocab_size:
        raise ValueError(f"vocab_size must be at least {initial_vocab_size}")
    if num_processes < 1:
        raise ValueError("num_processes must be positive")

    vocab: dict[int, bytes] = {index: token.encode("utf-8") for index, token in enumerate(specials)}
    byte_offset = len(vocab)
    vocab.update({byte_offset + value: _BYTE_TOKENS[value] for value in range(256)})

    pretoken_counts = _count_corpus_pretokens(input_path, specials, num_processes)
    words = [tuple(_BYTE_TOKENS[value] for value in pretoken) for pretoken in pretoken_counts]
    frequencies = list(pretoken_counts.values())

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_index, (word, frequency) in enumerate(zip(words, frequencies, strict=True)):
        for pair, occurrences in Counter(zip(word, word[1:], strict=False)).items():
            pair_counts[pair] += frequency * occurrences
            pair_to_words[pair].add(word_index)

    merges: list[tuple[bytes, bytes]] = []
    pair_heap = _pair_heap(pair_counts)
    while len(vocab) < vocab_size and pair_counts:
        best_pair = _pop_best_pair(pair_heap, pair_counts)
        merged_symbol = best_pair[0] + best_pair[1]
        affected_words = list(pair_to_words.pop(best_pair, ()))
        pair_deltas: Counter[tuple[bytes, bytes]] = Counter()

        for word_index in affected_words:
            old_word = words[word_index]
            frequency = frequencies[word_index]
            old_pairs = Counter(zip(old_word, old_word[1:], strict=False))
            for pair, occurrences in old_pairs.items():
                pair_deltas[pair] -= frequency * occurrences
                word_indexes = pair_to_words.get(pair)
                if word_indexes is not None:
                    word_indexes.discard(word_index)
                    if not word_indexes:
                        pair_to_words.pop(pair, None)

            new_word = _merge_pair(old_word, best_pair, merged_symbol)
            words[word_index] = new_word
            for pair, occurrences in Counter(zip(new_word, new_word[1:], strict=False)).items():
                pair_deltas[pair] += frequency * occurrences
                pair_to_words[pair].add(word_index)

        for pair, delta in pair_deltas.items():
            if delta == 0:
                continue
            new_frequency = pair_counts.get(pair, 0) + delta
            if new_frequency > 0:
                pair_counts[pair] = new_frequency
                heapq.heappush(pair_heap, _PairPriority(new_frequency, pair))
            else:
                pair_counts.pop(pair, None)

        # Bound stale-entry memory while keeping rebuilds amortized. The fixed
        # allowance avoids frequent rebuilds for tiny test corpora.
        if len(pair_heap) > 4 * len(pair_counts) + 1024:
            pair_heap = _pair_heap(pair_counts)

        merges.append(best_pair)
        vocab[len(vocab)] = merged_symbol

    return vocab, merges


class Tokenizer:
    """A byte-level BPE tokenizer with optional indivisible special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = _unique_in_order(special_tokens or [])
        self._max_special_token_length = max((len(token) for token in self.special_tokens), default=1)

        next_id = max(self.vocab, default=-1) + 1
        existing_tokens = set(self.vocab.values())
        for special_token in self.special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in existing_tokens:
                self.vocab[next_id] = encoded
                existing_tokens.add(encoded)
                next_id += 1

        self._token_to_id: dict[bytes, int] = {}
        for token_id in sorted(self.vocab):
            self._token_to_id.setdefault(self.vocab[token_id], token_id)

        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_regex = _special_pattern(self.special_tokens)
        self._special_to_id = {token: self._token_to_id[token.encode("utf-8")] for token in self.special_tokens}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load conventional GPT-2-style JSON vocabulary and merge files."""

        byte_decoder = {character: value for value, character in _bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as file:
            serialized_vocab = json.load(file)

        vocab: dict[int, bytes] = {}
        if all(isinstance(value, int) for value in serialized_vocab.values()):
            for encoded_token, token_id in serialized_vocab.items():
                vocab[token_id] = bytes(byte_decoder[character] for character in encoded_token)
        else:
            # Also accept {"id": [byte, ...]} for straightforward custom artifacts.
            for token_id, token_bytes in serialized_vocab.items():
                vocab[int(token_id)] = bytes(token_bytes)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                components = stripped.split()
                if len(components) != 2:
                    continue
                first, second = components
                merges.append(
                    (
                        bytes(byte_decoder[character] for character in first),
                        bytes(byte_decoder[character] for character in second),
                    )
                )
        return cls(vocab, merges, special_tokens)

    def save(
        self,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
    ) -> None:
        """Save vocabulary and merges in the conventional GPT-2 text formats."""

        byte_encoder = _bytes_to_unicode()

        def encode_bytes(value: bytes) -> str:
            return "".join(byte_encoder[byte] for byte in value)

        serialized_vocab = {encode_bytes(token): token_id for token_id, token in sorted(self.vocab.items())}
        Path(vocab_filepath).write_text(
            json.dumps(serialized_vocab, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with open(merges_filepath, "w", encoding="utf-8") as file:
            file.write("#version: 0.2\n")
            for first, second in self.merges:
                file.write(f"{encode_bytes(first)} {encode_bytes(second)}\n")

    def _encode_pretoken(self, pretoken: bytes) -> Iterator[int]:
        symbols = tuple(_BYTE_TOKENS[value] for value in pretoken)
        while len(symbols) > 1:
            ranked_pairs = (
                (self._merge_ranks[pair], pair)
                for pair in set(zip(symbols, symbols[1:], strict=False))
                if pair in self._merge_ranks
            )
            try:
                _, best_pair = min(ranked_pairs)
            except ValueError:
                break
            symbols = _merge_pair(symbols, best_pair, best_pair[0] + best_pair[1])

        for symbol in symbols:
            try:
                yield self._token_to_id[symbol]
            except KeyError as error:
                raise ValueError(f"vocabulary has no entry for byte token {symbol!r}") from error

    def _iter_token_units(self, text: str) -> Iterator[tuple[int, int, int | None]]:
        """Yield ordinary pre-token spans and indivisible special-token spans."""

        cursor = 0
        if self._special_regex is not None:
            for special_match in self._special_regex.finditer(text):
                for ordinary_match in GPT2_PRETOKEN_PATTERN.finditer(text, cursor, special_match.start()):
                    yield ordinary_match.start(), ordinary_match.end(), None
                yield (
                    special_match.start(),
                    special_match.end(),
                    self._special_to_id[special_match.group(0)],
                )
                cursor = special_match.end()

        for ordinary_match in GPT2_PRETOKEN_PATTERN.finditer(text, cursor):
            yield ordinary_match.start(), ordinary_match.end(), None

    def _encode_text(self, text: str) -> Iterator[int]:
        for start, end, special_id in self._iter_token_units(text):
            if special_id is None:
                yield from self._encode_pretoken(text[start:end].encode("utf-8"))
            else:
                yield special_id

    def encode(self, text: str) -> list[int]:
        """Encode a string to token IDs."""

        return list(self._encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode chunks without making chunk boundaries token boundaries."""

        pending = ""
        for text in iterable:
            if text == "":
                continue
            pending = pending + text if pending else text
            special_safe_end = max(0, len(pending) - self._max_special_token_length + 1)
            ready_units: deque[tuple[int, int, int | None]] = deque()
            emitted_end = 0

            # Future text can complete a special token in the guarded suffix
            # and can regroup the final two GPT-2 regex units (for example a
            # contraction or trailing whitespace), so retain both units.
            for start, end, special_id in self._iter_token_units(pending):
                if end > special_safe_end:
                    break
                ready_units.append((start, end, special_id))
                if len(ready_units) <= 2:
                    continue

                ready_start, ready_end, ready_special_id = ready_units.popleft()
                if ready_special_id is None:
                    yield from self._encode_pretoken(pending[ready_start:ready_end].encode("utf-8"))
                else:
                    yield ready_special_id
                emitted_end = ready_end

            if emitted_end:
                pending = pending[emitted_end:]

        if pending:
            yield from self._encode_text(pending)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs, replacing malformed UTF-8 with U+FFFD."""

        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


__all__ = ["GPT2_PRETOKEN_PATTERN", "Tokenizer", "train_bpe"]
