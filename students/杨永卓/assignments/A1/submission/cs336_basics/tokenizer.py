"""Byte-level BPE training and tokenization."""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import regex


# byte-level BPE 先用 GPT-2 风格正则预分词；pair merge 不会跨越 pre-token 边界。
GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@dataclass(frozen=True)
class _PairHeapEntry:
    count: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other: "_PairHeapEntry") -> bool:
        return (self.count, self.pair) > (other.count, other.pair)


def _merge_pair(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return tuple(merged)


def _update_pretoken_counts(
    counts: Counter[bytes], text: str, special_tokens: list[str], special_pattern: regex.Pattern | None
) -> None:
    # special token 必须保留完整边界，不能进入普通 BPE pair 统计。
    parts = special_pattern.split(text) if special_pattern is not None else (text,)
    for part in parts:
        if not part or part in special_tokens:
            continue
        counts.update(match.group().encode("utf-8") for match in GPT2_PRETOKEN_PATTERN.finditer(part))


def _pretoken_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    max_input_bytes: int | None = None,
) -> tuple[Counter[bytes], int]:
    counts: Counter[bytes] = Counter()
    special_pattern = None
    if special_tokens:
        alternatives = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        special_pattern = regex.compile(f"({alternatives})")
    if max_input_bytes is None:
        with open(input_path, encoding="utf-8") as source:
            text = source.read()
        _update_pretoken_counts(counts, text, special_tokens, special_pattern)
        return counts, len(text.encode("utf-8"))

    if max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be positive")

    # OWT 采用逐行、有上限的代表性采样，避免把完整大语料和 Counter 同时载入内存。
    sampled_bytes = 0
    with open(input_path, encoding="utf-8") as source:
        for line in source:
            encoded_line = line.encode("utf-8")
            remaining = max_input_bytes - sampled_bytes
            if remaining <= 0:
                break
            if len(encoded_line) > remaining:
                line = encoded_line[:remaining].decode("utf-8", errors="ignore")
                encoded_line = line.encode("utf-8")
            _update_pretoken_counts(counts, line, special_tokens, special_pattern)
            sampled_bytes += len(encoded_line)
            if sampled_bytes >= max_input_bytes:
                break
    return counts, sampled_bytes


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    progress: bool = False,
    max_input_bytes: int | None = None,
    stats: dict[str, int] | None = None,
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small for byte tokens and special tokens")

    pretoken_counts, sampled_bytes = _pretoken_counts(input_path, special_tokens, max_input_bytes)
    if stats is not None:
        stats.update(
            sampled_input_bytes=sampled_bytes,
            unique_pretokens=len(pretoken_counts),
        )
    # 相同 pre-token 只保留一份 token 序列，并用 frequency 记录其在语料中的出现次数。
    words = [tuple(bytes([value]) for value in pretoken) for pretoken in pretoken_counts]
    frequencies = list(pretoken_counts.values())

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_index, (word, frequency) in enumerate(zip(words, frequencies)):
        local_counts = Counter(zip(word, word[1:]))
        for pair, count in local_counts.items():
            pair_counts[pair] += count * frequency
            pair_to_words[pair].add(word_index)

    # 惰性最大堆避免每轮 merge 都扫描全部 pair；旧 entry 在弹出时按当前计数失效。
    pair_heap = [_PairHeapEntry(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    merges: list[tuple[bytes, bytes]] = []
    target_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(target_merges):
        while pair_heap:
            candidate = heapq.heappop(pair_heap)
            if pair_counts.get(candidate.pair) == candidate.count:
                break
        else:
            break
        best_pair = candidate.pair
        merges.append(best_pair)
        if progress and (len(merges) == 1 or len(merges) % 1000 == 0):
            print(
                f"BPE merges: {len(merges)}/{target_merges}; "
                f"best_count={pair_counts[best_pair]}",
                flush=True,
            )
        # 只重算包含 best_pair 的词，增量更新其余 pair 的频数，而非重新遍历整份语料。
        affected_words = tuple(pair_to_words.get(best_pair, ()))
        changed_pairs: set[tuple[bytes, bytes]] = set()
        for word_index in affected_words:
            old_word = words[word_index]
            frequency = frequencies[word_index]
            old_pairs = Counter(zip(old_word, old_word[1:]))
            for pair, count in old_pairs.items():
                changed_pairs.add(pair)
                pair_counts[pair] -= count * frequency
                pair_to_words[pair].discard(word_index)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            new_word = _merge_pair(old_word, best_pair)
            words[word_index] = new_word
            new_pairs = Counter(zip(new_word, new_word[1:]))
            for pair, count in new_pairs.items():
                changed_pairs.add(pair)
                pair_counts[pair] += count * frequency
                pair_to_words[pair].add(word_index)
        pair_to_words.pop(best_pair, None)
        for pair in changed_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(pair_heap, _PairHeapEntry(count, pair))

    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        existing_values = set(self.vocab.values())
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in existing_values:
                self.vocab[len(self.vocab)] = token_bytes
                existing_values.add(token_bytes)
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_pattern = None
        if self.special_tokens:
            alternatives = "|".join(
                regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            self.special_pattern = regex.compile(f"({alternatives})")

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        tokens = tuple(bytes([value]) for value in pretoken)
        while len(tokens) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(tokens, tokens[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            # 编码必须按训练时的 merge rank 依次应用，不能根据当前输入重新统计频率。
            _, best_pair = min(ranked_pairs)
            tokens = _merge_pair(tokens, best_pair)
        return [self.bytes_to_id[token] for token in tokens]

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group().encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return list(self._encode_ordinary(text))
        output: list[int] = []
        for part in self.special_pattern.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                output.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                output.extend(self._encode_ordinary(part))
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        # 先拼接全部 bytes 再整体 UTF-8 decode；单个 token 不一定恰好是完整 Unicode 字符。
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def save(self, path: str | os.PathLike) -> None:
        payload = {
            "vocab": {str(token_id): list(token_bytes) for token_id, token_bytes in self.vocab.items()},
            "merges": [[list(left), list(right)] for left, right in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Tokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {int(token_id): bytes(values) for token_id, values in payload["vocab"].items()}
        merges = [(bytes(left), bytes(right)) for left, right in payload["merges"]]
        return cls(vocab, merges, payload.get("special_tokens"))


BPEtokenizer = Tokenizer
run_train_bpe = train_bpe
