from __future__ import annotations

import json
import heapq
from collections.abc import Iterable, Iterator

import regex
from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path


# GPT-2 pre-tokenization pattern
PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class _ReversePair:
    """
    heapq is a min-heap. This wrapper makes lexicographically larger pairs sort
    earlier when pair counts tie, matching the assignment tie-break rule.
    """

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: "_ReversePair") -> bool:
        return self.pair > other.pair


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # 复制一份，避免修改调用者传入的 vocab 和 merges。
        self.vocab = dict(vocab)
        self.merges = list(merges)

        # bytes -> token id
        self.token_to_id: dict[bytes, int] = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # merge pair -> merge order
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}

        for rank, pair in enumerate(self.merges):
            # 如果意外存在重复 merge，保留第一次出现的顺序。
            self.merge_ranks.setdefault(pair, rank)

        # 去重，同时保留用户原本提供的顺序。
        self.special_tokens = list(
            dict.fromkeys(special_tokens or [])
        )

        if any(token == "" for token in self.special_tokens):
            raise ValueError("Special tokens must not be empty strings.")

        self.special_token_to_id: dict[str, int] = {}
        self.special_bytes_to_id: dict[bytes, int] = {}

        next_token_id = max(self.vocab, default=-1) + 1

        # 将缺失的 special token 加入 vocab。
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            token_id = self.token_to_id.get(special_bytes)

            if token_id is None:
                token_id = next_token_id
                next_token_id += 1

                self.vocab[token_id] = special_bytes
                self.token_to_id[special_bytes] = token_id

            self.special_token_to_id[special_token] = token_id
            self.special_bytes_to_id[special_bytes] = token_id

        # 长 special token 放在前面，避免短 token 抢先匹配。
        if self.special_tokens:
            ordered_special_tokens = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            alternatives = "|".join(
                regex.escape(token)
                for token in ordered_special_tokens
            )

            # 外层捕获组保证 regex.split 会保留 special token。
            self._special_pattern: regex.Pattern | None = regex.compile(
                f"({alternatives})"
            )
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike,
        merges_filepath: str | PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with Path(vocab_filepath).open("r", encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)

        vocab = {
            int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with Path(merges_filepath).open("r", encoding="utf-8") as merges_file:
            for line in merges_file:
                stripped = line.strip()
                if not stripped:
                    continue
                left_hex, right_hex = stripped.split()
                merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))

        return cls(vocab, merges, special_tokens)

    def _apply_bpe(
        self,
        pretoken_bytes: bytes,
    ) -> list[bytes]:
        """
        Apply BPE merges to one regex pre-token.
        """
        pieces = [
            bytes([byte_value])
            for byte_value in pretoken_bytes
        ]

        while len(pieces) >= 2:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            # 找当前相邻 pair 中训练顺序最早的 merge。
            for left, right in zip(pieces, pieces[1:]):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)

                if rank is not None and (
                    best_rank is None or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            # 一次合并该 pair 的所有非重叠出现位置。
            merged_pieces: list[bytes] = []
            index = 0

            while index < len(pieces):
                can_merge = (
                    index + 1 < len(pieces)
                    and pieces[index] == best_pair[0]
                    and pieces[index + 1] == best_pair[1]
                )

                if can_merge:
                    merged_pieces.append(
                        pieces[index] + pieces[index + 1]
                    )
                    index += 2
                else:
                    merged_pieces.append(pieces[index])
                    index += 1

            pieces = merged_pieces

        return pieces

    def _encode_ordinary_text(
        self,
        text: str,
    ) -> Iterator[int]:
        """
        Encode text that contains no recognized special tokens.
        """
        for match in PAT.finditer(text):
            pretoken_bytes = match.group(0).encode("utf-8")

            bpe_tokens = self._apply_bpe(pretoken_bytes)

            for token_bytes in bpe_tokens:
                try:
                    yield self.token_to_id[token_bytes]
                except KeyError as error:
                    raise KeyError(
                        f"Token bytes {token_bytes!r} are missing "
                        "from the vocabulary."
                    ) from error

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into token IDs.
        """
        if not text:
            return []

        if self._special_pattern is None:
            return list(self._encode_ordinary_text(text))

        token_ids: list[int] = []

        # split 的结果会交替出现：
        # 普通文本、special token、普通文本……
        for piece in self._special_pattern.split(text):
            if not piece:
                continue

            special_token_id = self.special_token_to_id.get(piece)

            if special_token_id is not None:
                token_ids.append(special_token_id)
            else:
                token_ids.extend(
                    self._encode_ordinary_text(piece)
                )

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        Lazily encode strings from an iterable.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(
        self,
        ids: Iterable[int],
    ) -> str:
        """
        Decode token IDs into a string.
        """
        decoded_bytes = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return decoded_bytes.decode(
            "utf-8",
            errors="replace",
        )


def train_bpe(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **_: object,
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab:
            Mapping from token ID to token bytes.

        merges:
            BPE merge pairs, ordered by creation time.
    """
    # 去重但保留 special token 的原始顺序。
    special_tokens = list(dict.fromkeys(special_tokens))

    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens must not be empty strings.")

    # 初始词表：全部 256 个单字节 token。
    vocab: dict[int, bytes] = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    vocab_values = set(vocab.values())
    next_token_id = 256

    # 将 special tokens 追加到词表。
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")

        if special_bytes not in vocab_values:
            vocab[next_token_id] = special_bytes
            vocab_values.add(special_bytes)
            next_token_id += 1

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is too small. "
            f"At least {len(vocab)} entries are required."
        )

    if vocab_size == len(vocab):
        return vocab, []

    text = Path(input_path).read_text(encoding="utf-8")

    # special token 只充当文本边界，不参与普通 BPE merge 统计。
    if special_tokens:
        ordered_special_tokens = sorted(
            special_tokens,
            key=len,
            reverse=True,
        )

        special_pattern = regex.compile(
            "("
            + "|".join(
                regex.escape(token)
                for token in ordered_special_tokens
            )
            + ")"
        )

        text_parts = special_pattern.split(text)
        special_token_set = set(special_tokens)
    else:
        text_parts = (text,)
        special_token_set: set[str] = set()

    # 避免反复创建相同的单字节 bytes。
    byte_tokens = tuple(
        bytes([byte_value])
        for byte_value in range(256)
    )

    # 每个 pre-token 的出现次数。
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

    for text_part in text_parts:
        if not text_part:
            continue

        if text_part in special_token_set:
            continue

        for match in PAT.finditer(text_part):
            pretoken_bytes = match.group(0).encode("utf-8")

            pretoken = tuple(
                byte_tokens[byte_value]
                for byte_value in pretoken_bytes
            )

            if pretoken:
                pretoken_counts[pretoken] += 1

    # 每个唯一 pre-token 只保存一次，另存它的语料频率。
    words: list[tuple[bytes, ...]] = list(
        pretoken_counts.keys()
    )

    word_frequencies: list[int] = [
        pretoken_counts[word]
        for word in words
    ]

    # pair_counts[pair]：
    # 该相邻 pair 在整个语料中的加权出现次数。
    pair_counts: Counter[
        tuple[bytes, bytes]
    ] = Counter()

    # pair_to_words[pair]：
    # 当前哪些唯一 pre-token 中包含这个 pair。
    pair_to_words: defaultdict[
        tuple[bytes, bytes],
        set[int],
    ] = defaultdict(set)

    for word_id, word in enumerate(words):
        frequency = word_frequencies[word_id]
        seen_pairs: set[tuple[bytes, bytes]] = set()

        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency
            seen_pairs.add(pair)

        for pair in seen_pairs:
            pair_to_words[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []

    pair_heap: list[
        tuple[
            int,
            _ReversePair,
            tuple[bytes, bytes],
        ]
    ] = [
        (-count, _ReversePair(pair), pair)
        for pair, count in pair_counts.items()
        if count > 0
    ]
    heapq.heapify(pair_heap)

    while len(vocab) < vocab_size and pair_counts:
        best_pair: tuple[bytes, bytes] | None = None
        best_count = 0

        # Lazy heap: pair counts are updated incrementally, so older heap
        # entries are discarded when they no longer match pair_counts.
        while pair_heap:
            neg_count, _, candidate_pair = heapq.heappop(pair_heap)
            candidate_count = -neg_count

            if pair_counts.get(candidate_pair, 0) == candidate_count:
                best_pair = candidate_pair
                best_count = candidate_count
                break

        if best_pair is None or best_count <= 0:
            break

        merged_token = best_pair[0] + best_pair[1]

        vocab[next_token_id] = merged_token
        next_token_id += 1

        merges.append(best_pair)

        # 只更新真正包含 best_pair 的 pre-token。
        affected_word_ids = tuple(
            pair_to_words.get(best_pair, ())
        )

        touched_pairs: set[
            tuple[bytes, bytes]
        ] = set()

        for word_id in affected_word_ids:
            old_word = words[word_id]
            frequency = word_frequencies[word_id]

            old_pairs = list(
                zip(old_word, old_word[1:])
            )

            touched_pairs.update(old_pairs)

            # 先移除该 word 对旧 pair 统计的贡献。
            for pair in old_pairs:
                pair_counts[pair] -= frequency

            for pair in set(old_pairs):
                containing_words = pair_to_words.get(pair)

                if containing_words is None:
                    continue

                containing_words.discard(word_id)

                if not containing_words:
                    pair_to_words.pop(pair, None)

            # 从左到右合并 best_pair 的所有非重叠出现。
            new_word_parts: list[bytes] = []
            position = 0

            while position < len(old_word):
                should_merge = (
                    position + 1 < len(old_word)
                    and old_word[position] == best_pair[0]
                    and old_word[position + 1] == best_pair[1]
                )

                if should_merge:
                    new_word_parts.append(
                        old_word[position]
                        + old_word[position + 1]
                    )
                    position += 2
                else:
                    new_word_parts.append(
                        old_word[position]
                    )
                    position += 1

            new_word = tuple(new_word_parts)
            words[word_id] = new_word

            new_pairs = list(
                zip(new_word, new_word[1:])
            )

            touched_pairs.update(new_pairs)

            # 加入该 word 对新 pair 统计的贡献。
            for pair in new_pairs:
                pair_counts[pair] += frequency

            for pair in set(new_pairs):
                pair_to_words[pair].add(word_id)

        # 清理已经不再出现的 pair。
        for pair in touched_pairs:
            if pair_counts.get(pair, 0) <= 0:
                pair_counts.pop(pair, None)
            else:
                heapq.heappush(
                    pair_heap,
                    (
                        -pair_counts[pair],
                        _ReversePair(pair),
                        pair,
                    ),
                )

    return vocab, merges
