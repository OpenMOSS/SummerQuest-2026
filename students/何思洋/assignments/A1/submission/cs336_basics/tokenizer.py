from __future__ import annotations

import os
import json
import heapq
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import regex as re


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _gpt2_byte_order() -> list[int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    bs.extend(b for b in range(256) if b not in bs)
    return bs


def _split_special_tokens(text: str, special_tokens: Iterable[str]) -> list[str]:
    specials = sorted(special_tokens, key=len, reverse=True)
    if not specials:
        return [text]
    pattern = "|".join(re.escape(token) for token in specials)
    return [part for part in re.split(pattern, text) if part]


def _split_text_with_specials(text: str, special_tokens: Iterable[str]) -> list[str]:
    specials = sorted(special_tokens, key=len, reverse=True)
    if not specials:
        return [text]
    pattern = "(" + "|".join(re.escape(token) for token in specials) + ")"
    return [part for part in re.split(pattern, text) if part]


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in _split_special_tokens(text, special_tokens):
        for match in re.finditer(GPT2_PRETOKEN_PATTERN, chunk):
            token_bytes = match.group(0).encode("utf-8")
            if token_bytes:
                counts[tuple(bytes([byte]) for byte in token_bytes)] += 1
    return counts


def _update_pretoken_counts(
    counts: Counter[tuple[bytes, ...]], text: str, special_tokens: list[str]
) -> None:
    for chunk in _split_special_tokens(text, special_tokens):
        for match in re.finditer(GPT2_PRETOKEN_PATTERN, chunk):
            token_bytes = match.group(0).encode("utf-8")
            if token_bytes:
                counts[tuple(bytes([byte]) for byte in token_bytes)] += 1


def _pretoken_counts_from_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    progress_callback: Callable[[str, dict[str, int]], None] | None = None,
    progress_every_bytes: int = 256 * 1024 * 1024,
) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    bytes_read = 0
    next_report = progress_every_bytes
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            _update_pretoken_counts(counts, line, special_tokens)
            bytes_read += len(line.encode("utf-8"))
            if progress_callback is not None and bytes_read >= next_report:
                progress_callback(
                    "pretokenize",
                    {"bytes_read": bytes_read, "unique_pretokens": len(counts)},
                )
                while next_report <= bytes_read:
                    next_report += progress_every_bytes
    if progress_callback is not None:
        progress_callback(
            "pretokenize_done",
            {"bytes_read": bytes_read, "unique_pretokens": len(counts)},
        )
    return counts


def _pair_counts(word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, frequency in word_counts.items():
        for pair in _word_pairs(word):
            counts[pair] += frequency
    return counts


def _pair_heap(pair_counts: Counter[tuple[bytes, bytes]]) -> list[tuple[int, tuple[bytes, bytes]]]:
    return [(-count, pair) for pair, count in pair_counts.items()]


def _best_pair(
    heap: list[tuple[int, tuple[bytes, bytes]]],
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes] | None:
    while heap:
        neg_count, pair = heapq.heappop(heap)
        count = -neg_count
        if pair_counts.get(pair) != count:
            continue

        candidates = [pair]
        while heap and -heap[0][0] == count:
            other_neg_count, other_pair = heapq.heappop(heap)
            if pair_counts.get(other_pair) == -other_neg_count:
                candidates.append(other_pair)

        best = max(candidates)
        for candidate in candidates:
            if candidate != best:
                heapq.heappush(heap, (-count, candidate))
        return best
    return None


def _word_pairs(word: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    return list(zip(word, word[1:]))


def _pair_index(word_counts: Counter[tuple[bytes, ...]]) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for word in word_counts:
        for pair in _word_pairs(word):
            index.setdefault(pair, set()).add(word)
    return index


def _decrement_pair(
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair: tuple[bytes, bytes],
    word: tuple[bytes, ...],
    frequency: int,
) -> None:
    pair_counts[pair] -= frequency
    if pair_counts[pair] <= 0:
        del pair_counts[pair]
    words = pair_to_words.get(pair)
    if words is not None:
        words.discard(word)
        if not words:
            del pair_to_words[pair]


def _increment_pair(
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair: tuple[bytes, bytes],
    word: tuple[bytes, ...],
    frequency: int,
) -> None:
    pair_counts[pair] += frequency
    pair_to_words.setdefault(pair, set()).add(word)


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_token_ids = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self.token_to_id
        }

    def _apply_merges(self, token_bytes: bytes) -> tuple[bytes, ...]:
        parts = tuple(bytes([byte]) for byte in token_bytes)
        if len(parts) < 2:
            return parts

        while True:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(parts, parts[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                return parts
            _, best_pair = min(ranked_pairs)
            parts = _merge_word(parts, best_pair)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _split_text_with_specials(text, self.special_tokens):
            special_id = self.special_token_ids.get(chunk)
            if special_id is not None:
                ids.append(special_id)
                continue
            for match in re.finditer(GPT2_PRETOKEN_PATTERN, chunk):
                for token in self._apply_merges(match.group(0).encode("utf-8")):
                    ids.append(self.token_to_id[token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")


def make_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def save_tokenizer(
    path: str | os.PathLike,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> None:
    payload = {
        "vocab": {str(token_id): token.hex() for token_id, token in vocab.items()},
        "merges": [[left.hex(), right.hex()] for left, right in merges],
        "special_tokens": special_tokens or [],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_tokenizer_spec(path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    vocab = {int(token_id): bytes.fromhex(token) for token_id, token in payload["vocab"].items()}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in payload["merges"]]
    return vocab, merges, payload.get("special_tokens", [])


def load_tokenizer(path: str | os.PathLike) -> BPETokenizer:
    vocab, merges, special_tokens = load_tokenizer_spec(path)
    return make_tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    progress_callback: Callable[[str, dict[str, int]], None] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for byte in _gpt2_byte_order():
        vocab[len(vocab)] = bytes([byte])

    if os.path.getsize(input_path) <= 512 * 1024 * 1024:
        with open(input_path, encoding="utf-8") as f:
            word_counts = _pretoken_counts(f.read(), special_tokens)
    else:
        word_counts = _pretoken_counts_from_file(input_path, special_tokens, progress_callback=progress_callback)
    pair_counts = _pair_counts(word_counts)
    heap = _pair_heap(pair_counts)
    heapq.heapify(heap)
    pair_to_words = _pair_index(word_counts)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = _best_pair(heap, pair_counts)
        if best_pair is None:
            break
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_words = list(pair_to_words.get(best_pair, set()))
        merged_word_counts: Counter[tuple[bytes, ...]] = Counter()

        for word in affected_words:
            frequency = word_counts.pop(word, 0)
            if frequency == 0:
                continue
            for pair in _word_pairs(word):
                _decrement_pair(pair_counts, pair_to_words, pair, word, frequency)
                if pair in pair_counts:
                    heapq.heappush(heap, (-pair_counts[pair], pair))
            merged_word_counts[_merge_word(word, best_pair)] += frequency

        for word, frequency in merged_word_counts.items():
            word_counts[word] += frequency
            for pair in _word_pairs(word):
                _increment_pair(pair_counts, pair_to_words, pair, word, frequency)
                heapq.heappush(heap, (-pair_counts[pair], pair))

        if progress_callback is not None and (len(merges) == 1 or len(merges) % 1000 == 0):
            progress_callback(
                "merge",
                {
                    "vocab_size": len(vocab),
                    "num_merges": len(merges),
                    "num_pairs": len(pair_counts),
                    "num_pretokens": len(word_counts),
                },
            )

    return vocab, merges
