from __future__ import annotations

import os
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable, Iterator

import regex as re


GPT2_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
STREAMING_BPE_THRESHOLD_BYTES = 64 * 1024 * 1024


def _special_pattern(special_tokens: list[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    alternatives = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return re.compile(f"({alternatives})")


def _split_on_specials(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    pattern = _special_pattern(special_tokens)
    if pattern is None:
        if text:
            yield text, False
        return

    start = 0
    for match in pattern.finditer(text):
        if match.start() > start:
            yield text[start : match.start()], False
        yield match.group(0), True
        start = match.end()
    if start < len(text):
        yield text[start:], False


def _pretoken_bytes(text: str) -> Iterator[bytes]:
    for match in GPT2_PRETOKEN_PATTERN.finditer(text):
        yield match.group(0).encode("utf-8")


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for token in special_tokens:
        if token.encode("utf-8") not in vocab.values():
            vocab[len(vocab)] = token.encode("utf-8")

    def update_word_counts(text: str, word_counts: Counter[tuple[int, ...]]) -> None:
        for piece, is_special in _split_on_specials(text, special_tokens):
            if is_special:
                continue
            for token_bytes in _pretoken_bytes(piece):
                word_counts[tuple(token_bytes)] += 1

    word_counts: Counter[tuple[int, ...]] = Counter()
    input_size = os.path.getsize(input_path)
    with open(input_path, encoding="utf-8") as f:
        if input_size <= STREAMING_BPE_THRESHOLD_BYTES:
            update_word_counts(f.read(), word_counts)
        else:
            for line in f:
                update_word_counts(line, word_counts)

    words = [list(word) for word in word_counts]
    counts = [word_counts[tuple(word)] for word in words]
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

    for word_idx, word in enumerate(words):
        count = counts[word_idx]
        for pair in zip(word, word[1:], strict=False):
            pair_counts[pair] += count
            pair_to_words[pair].add(word_idx)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]))
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        affected_words = list(pair_to_words.pop(best_pair, set()))
        for word_idx in affected_words:
            word = words[word_idx]
            count = counts[word_idx]
            if not any(pair == best_pair for pair in zip(word, word[1:], strict=False)):
                continue

            for pair in zip(word, word[1:], strict=False):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    pair_to_words.pop(pair, None)
                else:
                    pair_to_words[pair].discard(word_idx)

            merged: list[int] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[word_idx] = merged

            for pair in zip(merged, merged[1:], strict=False):
                pair_counts[pair] += count
                pair_to_words[pair].add(word_idx)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.byte_to_id = {value: key for key, value in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_bytes_to_id = {
            token.encode("utf-8"): self.byte_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self.byte_to_id
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        import json

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(idx): bytes.fromhex(value) for idx, value in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                left, right = line.rstrip("\n").split("\t")
                merges.append((bytes.fromhex(left), bytes.fromhex(right)))
        return cls(vocab, merges, special_tokens)

    def _encode_bytes(self, token_bytes: bytes) -> list[int]:
        parts = tuple(bytes([byte]) for byte in token_bytes)
        if not parts:
            return []

        while len(parts) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair) for pair in zip(parts, parts[1:], strict=False) if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs)
            parts = _merge_word(parts, best_pair)
        return [self.byte_to_id[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece, is_special in _split_on_specials(text, self.special_tokens):
            if is_special:
                ids.append(self.special_bytes_to_id[piece.encode("utf-8")])
                continue
            for token_bytes in _pretoken_bytes(piece):
                ids.extend(self._encode_bytes(token_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self.vocab[idx] for idx in ids)
        return data.decode("utf-8", errors="replace")
