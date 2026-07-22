from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import regex


PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
)


def _special_pattern(special_tokens: list[str]) -> regex.Pattern | None:
    if not special_tokens:
        return None
    alternatives = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return regex.compile(f"({alternatives})")


def _ordinary_pretokens(text: str) -> Iterator[bytes]:
    for match in PRETOKEN_PATTERN.finditer(text):
        yield match.group().encode("utf-8")


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    output: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            output.append(pair[0] + pair[1])
            i += 2
        else:
            output.append(word[i])
            i += 1
    return tuple(output)


def _find_chunk_boundaries(file, desired_num_chunks: int, split_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if desired_num_chunks <= 1 or not split_token:
        return [0, file_size]
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [index * chunk_size for index in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    for index in range(1, len(boundaries) - 1):
        position = boundaries[index]
        file.seek(position)
        remainder = b""
        found_boundary = False
        while position < file_size:
            block = file.read(1 << 20)
            if not block:
                break
            combined = remainder + block
            found = combined.find(split_token)
            if found >= 0:
                boundaries[index] = position - len(remainder) + found
                found_boundary = True
                break
            remainder = combined[-max(0, len(split_token) - 1) :]
            position += len(block)
        if not found_boundary:
            boundaries[index] = file_size
    return sorted(set(boundaries))


def _count_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: tuple[str, ...],
) -> Counter[bytes]:
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")
    splitter = _special_pattern(list(special_tokens))
    pieces = splitter.split(text) if splitter else [text]
    special_set = set(special_tokens)
    counts: Counter[bytes] = Counter()
    for piece in pieces:
        if piece not in special_set:
            counts.update(_ordinary_pretokens(piece))
    return counts


class _ReversePair:
    """Make heapq select the lexicographically greatest pair when counts tie."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_ReversePair") -> bool:
        return self.pair > other.pair


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small for byte tokens and special tokens")

    input_path = str(input_path)
    num_processes = int(kwargs.get("num_processes", min(8, os.cpu_count() or 1)))
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b""
    with open(input_path, "rb") as file:
        boundaries = _find_chunk_boundaries(file, num_processes, split_token)
    jobs = [(input_path, start, end, tuple(special_tokens)) for start, end in zip(boundaries, boundaries[1:])]
    if len(jobs) == 1:
        pretoken_counts = _count_chunk(*jobs[0])
    else:
        pretoken_counts: Counter[bytes] = Counter()
        with ProcessPoolExecutor(max_workers=min(num_processes, len(jobs))) as executor:
            for partial in executor.map(_count_chunk_star, jobs):
                pretoken_counts.update(partial)

    word_freq: Counter[tuple[bytes, ...]] = Counter()
    byte_tokens = tuple(bytes([byte]) for byte in range(256))
    for pretoken, frequency in pretoken_counts.items():
        word_freq[tuple(byte_tokens[byte] for byte in pretoken)] = frequency

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, frequency in word_freq.items():
        pairs = Counter(zip(word, word[1:]))
        for pair, occurrences in pairs.items():
            pair_counts[pair] += frequency * occurrences
            pair_words[pair].add(word)
    pair_heap = [(-count, _ReversePair(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    vocab = {index: bytes([index]) for index in range(256)}
    for token in special_tokens:
        encoded = token.encode("utf-8")
        if encoded not in vocab.values():
            vocab[len(vocab)] = encoded
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and pair_heap:
        while pair_heap:
            negative_count, _, candidate = heapq.heappop(pair_heap)
            if pair_counts.get(candidate, 0) == -negative_count:
                best_pair = candidate
                break
        else:
            break
        affected_words = list(pair_words[best_pair])
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

        changed_pairs: set[tuple[bytes, bytes]] = set()
        for old_word in affected_words:
            frequency = word_freq.get(old_word, 0)
            if not frequency:
                continue
            new_word = _merge_word(old_word, best_pair)
            if new_word == old_word:
                continue
            old_pairs = Counter(zip(old_word, old_word[1:]))
            new_pairs = Counter(zip(new_word, new_word[1:]))
            for pair, occurrences in old_pairs.items():
                pair_counts[pair] -= frequency * occurrences
                pair_words[pair].discard(old_word)
                changed_pairs.add(pair)
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
            for pair, occurrences in new_pairs.items():
                pair_counts[pair] += frequency * occurrences
                pair_words[pair].add(new_word)
                changed_pairs.add(pair)
            del word_freq[old_word]
            word_freq[new_word] += frequency

        pair_words.pop(best_pair, None)
        for pair in changed_pairs:
            if pair in pair_counts:
                heapq.heappush(pair_heap, (-pair_counts[pair], _ReversePair(pair), pair))

    return vocab, merges


def _count_chunk_star(job: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    return _count_chunk(*job)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = list(special_tokens or [])
        next_token_id = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self.token_to_id:
                self.vocab[next_token_id] = encoded
                self.token_to_id[encoded] = next_token_id
                next_token_id += 1
        self.special_to_id = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }
        self.special_splitter = _special_pattern(self.special_tokens)
        self._max_special_token_length = max((len(token) for token in self.special_tokens), default=1)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        import json

        with open(vocab_filepath, encoding="utf-8") as file:
            raw_vocab = json.load(file)
        vocab = {int(index): bytes.fromhex(token_hex) for index, token_hex in raw_vocab.items()}
        with open(merges_filepath, encoding="utf-8") as file:
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in (line.split() for line in file if line.strip())]
        return cls(vocab, merges, special_tokens)

    @lru_cache(maxsize=100_000)
    def _encode_pretoken(self, pretoken: bytes) -> tuple[int, ...]:
        parts = tuple(bytes([byte]) for byte in pretoken)
        while len(parts) > 1:
            ranked = ((self.merge_ranks[pair], pair) for pair in zip(parts, parts[1:]) if pair in self.merge_ranks)
            try:
                _, best_pair = min(ranked)
            except ValueError:
                break
            parts = _merge_word(parts, best_pair)
        return tuple(self.token_to_id[part] for part in parts)

    def _iter_token_units(self, text: str) -> Iterator[tuple[int, int, int | None]]:
        """Yield ordinary pre-token spans and indivisible special-token spans."""

        cursor = 0
        if self.special_splitter is not None:
            for special_match in self.special_splitter.finditer(text):
                for ordinary_match in PRETOKEN_PATTERN.finditer(text, cursor, special_match.start()):
                    yield ordinary_match.start(), ordinary_match.end(), None
                yield (
                    special_match.start(),
                    special_match.end(),
                    self.special_to_id[special_match.group()],
                )
                cursor = special_match.end()

        for ordinary_match in PRETOKEN_PATTERN.finditer(text, cursor):
            yield ordinary_match.start(), ordinary_match.end(), None

    def _encode_text(self, text: str) -> Iterator[int]:
        for start, end, special_id in self._iter_token_units(text):
            if special_id is None:
                yield from self._encode_pretoken(text[start:end].encode("utf-8"))
            else:
                yield special_id

    def encode(self, text: str) -> list[int]:
        return list(self._encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending = ""
        for text in iterable:
            if not text:
                continue
            pending = pending + text if pending else text
            special_safe_end = max(0, len(pending) - self._max_special_token_length + 1)
            ready_units: deque[tuple[int, int, int | None]] = deque()
            emitted_end = 0

            # A later chunk may complete a special token or regroup the final
            # GPT-2 regex units (for example a contraction or whitespace run).
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

    def decode(self, ids: list[int]) -> str:
        try:
            encoded = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return encoded.decode("utf-8", errors="replace")


def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_prefix: str | Path) -> None:
    import json

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(prefix.with_suffix(".vocab.json"), "w", encoding="utf-8") as file:
        json.dump({str(index): token.hex() for index, token in vocab.items()}, file, indent=2)
    with open(prefix.with_suffix(".merges.txt"), "w", encoding="utf-8") as file:
        for first, second in merges:
            file.write(f"{first.hex()} {second.hex()}\n")
