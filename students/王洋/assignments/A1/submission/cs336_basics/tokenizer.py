"""Byte-level BPE training and tokenization.

The implementation in this module deliberately keeps the external representation
of a token as ``bytes``.  This makes all 256 input bytes representable and avoids
coupling the tokenizer itself to GPT-2's printable byte-to-Unicode serialization.
"""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any
import regex


# This is the pre-tokenization expression used by GPT-2.  In particular, the
# literal optional space in the middle alternatives is intentional.
GPT2_PRETOKEN_PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
_CONTRACTIONS = ("'s", "'d", "'m", "'t", "'ll", "'ve", "'re")


def _unique_special_tokens(special_tokens: Iterable[str] | None) -> list[str]:
    """Return non-empty special tokens in input order, without duplicates."""

    result: list[str] = []
    seen: set[str] = set()
    for token in special_tokens or ():
        if not isinstance(token, str):
            raise TypeError("special tokens must be strings")
        if token == "":
            raise ValueError("the empty string cannot be a special token")
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _compile_special_pattern(special_tokens: list[str]) -> Any | None:
    """Compile a leftmost-longest special-token matcher.

    ``regex`` alternations choose the first alternative at a shared start
    position, so ordering longer strings first gives the required behavior for
    overlapping tokens such as ``<eot>`` and ``<eot><eot>``.
    """

    if not special_tokens:
        return None
    ordered = sorted(enumerate(special_tokens), key=lambda item: (-len(item[1]), item[0]))
    return regex.compile("|".join(regex.escape(token) for _, token in ordered))


def _count_pretokens(text: str, counts: Counter[bytes]) -> None:
    """Add the byte sequences of all GPT-2 pre-tokens in ``text`` to ``counts``."""

    for match in GPT2_PRETOKEN_PATTERN.finditer(text):
        # ``bytes`` has exactly the ordering and iteration semantics needed by
        # the rest of the trainer, while using far less memory than a tuple of
        # Python integers for every unique pre-token.  This matters on OWT,
        # where the tuple representation alone can consume tens of gigabytes.
        counts[match.group(0).encode("utf-8")] += 1


def _count_pretokens_from_file(
    input_path: str | os.PathLike[str],
    counts: Counter[bytes],
    special_pattern: Any | None,
    *,
    chunk_chars: int = 1024 * 1024,
) -> None:
    """Count pre-tokens without materializing a special-delimited corpus.

    A complete special token is a hard boundary and contributes no ordinary
    pre-tokens.  Keeping only the unfinished segment after the last such
    boundary makes the result identical to splitting a fully-read string while
    bounding text memory by roughly one document plus one input chunk.  The
    no-special case retains the simple whole-file path because there is no
    guaranteed semantic boundary at which an arbitrary regex token can be
    flushed.
    """

    with Path(input_path).open(encoding="utf-8") as corpus_file:
        if special_pattern is None:
            _count_pretokens(corpus_file.read(), counts)
            return

        pending = ""
        while chunk := corpus_file.read(chunk_chars):
            pending += chunk
            cursor = 0
            for match in special_pattern.finditer(pending):
                _count_pretokens(pending[cursor : match.start()], counts)
                cursor = match.end()
            if cursor:
                pending = pending[cursor:]
        _count_pretokens(pending, counts)


class _PairPriority:
    """A max-priority heap item implemented for Python's min-heap.

    BPE chooses the pair with greatest frequency, breaking ties by the
    lexicographically greatest pair of byte strings.
    """

    __slots__ = ("count", "pair", "lexical_pair")

    def __init__(self, count: int, pair: tuple[int, int], lexical_pair: tuple[bytes, bytes]) -> None:
        self.count = count
        self.pair = pair
        self.lexical_pair = lexical_pair

    def __lt__(self, other: _PairPriority) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.lexical_pair > other.lexical_pair


def _adjacent_pair_counts(word: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(word, word[1:]))


def _merge_pair(word: list[int], pair: tuple[int, int], merged_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` from left to right."""

    left, right = pair
    merged: list[int] = []
    index = 0
    while index < len(word):
        if index + 1 < len(word) and word[index] == left and word[index + 1] == right:
            merged.append(merged_id)
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return merged


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a deterministic byte-level BPE tokenizer.

    The initial vocabulary contains all 256 one-byte tokens followed by the
    requested special tokens.  Special-token occurrences divide the corpus and
    are excluded from pair statistics, so no learned token can cross one of
    those boundaries.
    """

    specials = _unique_special_tokens(special_tokens)
    minimum_vocab_size = 256 + len(specials)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size} (256 byte tokens plus the special tokens)")

    vocab: dict[int, bytes] = {byte: bytes((byte,)) for byte in range(256)}
    token_bytes: list[bytes] = [bytes((byte,)) for byte in range(256)]
    for special in specials:
        encoded = special.encode("utf-8")
        vocab[len(vocab)] = encoded
        token_bytes.append(encoded)

    # Counting unique pre-tokens once, with a corpus-frequency multiplier, is
    # both faster and substantially smaller than retaining every occurrence.
    special_pattern = _compile_special_pattern(specials)
    pretoken_counts: Counter[bytes] = Counter()
    _count_pretokens_from_file(input_path, pretoken_counts, special_pattern)

    words = [list(word) for word in pretoken_counts]
    frequencies = list(pretoken_counts.values())
    del pretoken_counts

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for word_id, (word, frequency) in enumerate(zip(words, frequencies, strict=True)):
        for pair, occurrences in _adjacent_pair_counts(word).items():
            pair_counts[pair] += frequency * occurrences
            pair_to_words[pair].add(word_id)

    heap: list[_PairPriority] = [
        _PairPriority(count, pair, (token_bytes[pair[0]], token_bytes[pair[1]])) for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        # Counts change incrementally.  Old heap entries are discarded lazily.
        while heap:
            candidate = heapq.heappop(heap)
            if pair_counts.get(candidate.pair) == candidate.count:
                best_pair = candidate.pair
                break
        else:
            break

        left_bytes = token_bytes[best_pair[0]]
        right_bytes = token_bytes[best_pair[1]]
        merged_bytes = left_bytes + right_bytes
        merged_id = len(vocab)
        vocab[merged_id] = merged_bytes
        token_bytes.append(merged_bytes)
        merges.append((left_bytes, right_bytes))

        affected_words = tuple(pair_to_words.get(best_pair, ()))
        dirty_pairs: set[tuple[int, int]] = set()
        for word_id in affected_words:
            word = words[word_id]
            frequency = frequencies[word_id]

            for pair, occurrences in _adjacent_pair_counts(word).items():
                pair_counts[pair] -= frequency * occurrences
                dirty_pairs.add(pair)
                holders = pair_to_words.get(pair)
                if holders is not None:
                    holders.discard(word_id)
                    if not holders:
                        del pair_to_words[pair]
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            new_word = _merge_pair(word, best_pair, merged_id)
            words[word_id] = new_word
            for pair, occurrences in _adjacent_pair_counts(new_word).items():
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency * occurrences
                pair_to_words[pair].add(word_id)
                dirty_pairs.add(pair)

        for pair in dirty_pairs:
            count = pair_counts.get(pair)
            if count:
                heapq.heappush(
                    heap,
                    _PairPriority(count, pair, (token_bytes[pair[0]], token_bytes[pair[1]])),
                )

    return vocab, merges


@lru_cache(maxsize=1)
def _gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse of GPT-2's printable byte-to-Unicode mapping."""

    visible_bytes = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = visible_bytes[:]
    unicode_values = visible_bytes[:]
    offset = 0
    for byte in range(256):
        if byte not in visible_bytes:
            byte_values.append(byte)
            unicode_values.append(256 + offset)
            offset += 1
    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, unicode_values, strict=True)}


def _deserialize_gpt2_token(token: str) -> bytes:
    """Decode one token from GPT-2's JSON/text serialization."""

    decoder = _gpt2_byte_decoder()
    result = bytearray()
    for character in token:
        byte = decoder.get(character)
        if byte is None:
            result.extend(character.encode("utf-8"))
        else:
            result.append(byte)
    return bytes(result)


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
        self.special_tokens = _unique_special_tokens(special_tokens)

        next_id = max(self.vocab, default=-1) + 1
        existing_ids: dict[bytes, int] = {}
        for token_id, token in sorted(self.vocab.items()):
            if not isinstance(token_id, int) or not isinstance(token, bytes):
                raise TypeError("vocab must map integer token IDs to bytes")
            existing_ids.setdefault(token, token_id)

        self._special_to_id: dict[str, int] = {}
        for special in self.special_tokens:
            encoded = special.encode("utf-8")
            token_id = existing_ids.get(encoded)
            if token_id is None:
                token_id = next_id
                next_id += 1
                self.vocab[token_id] = encoded
                existing_ids[encoded] = token_id
            self._special_to_id[special] = token_id

        self._bytes_to_id = existing_ids
        self._special_pattern = _compile_special_pattern(self.special_tokens)
        self._longest_special = max(map(len, self.special_tokens), default=0)

        # Convert byte-pair ranks to integer-pair actions once.  Encoding then
        # performs only integer dictionary lookups in its inner loop.
        self._merge_actions: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right) in enumerate(self.merges):
            left_id = self._bytes_to_id.get(left)
            right_id = self._bytes_to_id.get(right)
            merged_id = self._bytes_to_id.get(left + right)
            if left_id is None or right_id is None or merged_id is None:
                raise ValueError(f"merge {(left, right)!r} refers to a token missing from the vocabulary")
            self._merge_actions.setdefault((left_id, right_id), (rank, merged_id))

        try:
            self._byte_ids = tuple(self._bytes_to_id[bytes((byte,))] for byte in range(256))
        except KeyError as error:
            raise ValueError("a byte-level tokenizer vocabulary must contain all 256 one-byte tokens") from error

        # Pre-tokens repeat heavily in natural language.  A bounded cache gives
        # most of that speedup without allowing streaming memory use to grow
        # with the size of the input corpus.
        self._encode_bytes_cached = lru_cache(maxsize=8192)(self._encode_bytes_uncached)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Construct a tokenizer from GPT-2-style vocabulary and merge files."""

        with Path(vocab_filepath).open(encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)
        if not isinstance(serialized_vocab, dict):
            raise ValueError("the vocabulary file must contain a JSON object")

        vocab: dict[int, bytes] = {}
        for serialized_token, token_id in serialized_vocab.items():
            if not isinstance(serialized_token, str) or not isinstance(token_id, int):
                raise ValueError("the vocabulary must map serialized token strings to integer IDs")
            vocab[token_id] = _deserialize_gpt2_token(serialized_token)

        merges: list[tuple[bytes, bytes]] = []
        with Path(merges_filepath).open(encoding="utf-8") as merges_file:
            for line in merges_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                pieces = stripped.split()
                if len(pieces) != 2:
                    continue
                merges.append((_deserialize_gpt2_token(pieces[0]), _deserialize_gpt2_token(pieces[1])))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_bytes_uncached(self, piece: bytes) -> tuple[int, ...]:
        if not piece:
            return ()

        tokens = [self._byte_ids[byte] for byte in piece]
        while len(tokens) > 1:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            merged_id: int | None = None
            for pair in zip(tokens, tokens[1:]):
                action = self._merge_actions.get(pair)
                if action is not None and (best_rank is None or action[0] < best_rank):
                    best_rank = action[0]
                    best_pair = pair
                    merged_id = action[1]

            if best_pair is None or merged_id is None:
                break

            tokens = _merge_pair(tokens, best_pair, merged_id)
        return tuple(tokens)

    def _iter_atoms(self, text: str) -> Iterator[tuple[int, int, int | None]]:
        """Yield ``(start, end, special_id)`` atoms covering ``text``."""

        def ordinary_atoms(start: int, end: int) -> Iterator[tuple[int, int, None]]:
            for pretoken in GPT2_PRETOKEN_PATTERN.finditer(text, start, end):
                yield pretoken.start(), pretoken.end(), None

        if self._special_pattern is None:
            yield from ordinary_atoms(0, len(text))
            return

        cursor = 0
        for special_match in self._special_pattern.finditer(text):
            yield from ordinary_atoms(cursor, special_match.start())
            special = special_match.group(0)
            yield special_match.start(), special_match.end(), self._special_to_id[special]
            cursor = special_match.end()
        yield from ordinary_atoms(cursor, len(text))

    def _encode_atom(self, text: str, atom: tuple[int, int, int | None]) -> Iterator[int]:
        start, end, special_id = atom
        if special_id is not None:
            yield special_id
        else:
            yield from self._encode_bytes_cached(text[start:end].encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into token IDs."""

        if not isinstance(text, str):
            raise TypeError("Tokenizer.encode expects a string")
        encoded: list[int] = []
        for atom in self._iter_atoms(text):
            encoded.extend(self._encode_atom(text, atom))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode text chunks without loading the complete input into memory.

        The final ordinary pre-token and enough characters to contain a partial
        special token are carried to the next chunk.  Consequently, arbitrary
        chunk boundaries do not change tokenization.
        """

        buffer = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("Tokenizer.encode_iterable expects an iterable of strings")
            if not chunk:
                continue
            buffer += chunk
            atoms = list(self._iter_atoms(buffer))
            if not atoms:
                continue

            retain_from = len(buffer)

            # An ordinary pre-token can continue indefinitely in the next
            # chunk (a word, digit run, punctuation run, or whitespace run).
            if atoms[-1][2] is None:
                retain_from = atoms[-1][0]

            # The apostrophe alternatives are the one part of the GPT-2
            # expression that can retroactively combine multiple previously
            # separate atoms (for example, chunks ending in ``'`` then ``l``
            # then ``l``).  Keep an incomplete contraction prefix intact.
            for prefix_length in (1, 2):
                if prefix_length > len(buffer):
                    continue
                suffix = buffer[-prefix_length:]
                if any(contraction.startswith(suffix) for contraction in _CONTRACTIONS):
                    suffix_start = len(buffer) - prefix_length
                    for atom in atoms:
                        if atom[1] > suffix_start:
                            retain_from = min(retain_from, atom[0])
                            break
                    break

            # A suffix may be the beginning of a special token.  Retain whole
            # atoms intersecting that suffix so it can be reconsidered later.
            if self._longest_special > 1:
                suffix_start = max(0, len(buffer) - (self._longest_special - 1))
                for atom in atoms:
                    if atom[1] > suffix_start:
                        retain_from = min(retain_from, atom[0])
                        break

            for atom in atoms:
                if atom[1] > retain_from:
                    break
                yield from self._encode_atom(buffer, atom)
            buffer = buffer[retain_from:]

        if buffer:
            for atom in self._iter_atoms(buffer):
                yield from self._encode_atom(buffer, atom)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs, replacing malformed UTF-8 with U+FFFD."""

        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


__all__ = ["GPT2_PRETOKEN_PATTERN", "Tokenizer", "train_bpe"]
