from collections.abc import Iterable, Iterator

import regex

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import os

import regex


GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        self.special_tokens = special_tokens or []

        self.special_token_to_id: dict[str, int] = {}

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                raise ValueError(
                    f"Special token {special_token!r} is not in the vocabulary"
                )

            self.special_token_to_id[special_token] = (
                self.token_to_id[token_bytes]
            )

        # 长的 special token 放在前面，防止短 token 抢先匹配。
        sorted_special_tokens = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        if sorted_special_tokens:
            alternatives = "|".join(
                regex.escape(token)
                for token in sorted_special_tokens
            )

            self.special_token_pattern = regex.compile(
                f"({alternatives})"
            )
        else:
            self.special_token_pattern = None

        # 相同的 pre-token 经常重复出现，缓存其 BPE 结果。
        self.bpe_cache: dict[bytes, tuple[bytes, ...]] = {}

    def _apply_bpe(self, token_bytes: bytes) -> tuple[bytes, ...]:
        if token_bytes in self.bpe_cache:
            return self.bpe_cache[token_bytes]

        pieces = [bytes([byte]) for byte in token_bytes]

        while len(pieces) >= 2:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            for left, right in zip(pieces, pieces[1:]):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)

                if rank is not None and (
                    best_rank is None or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            # 当前相邻 token 中，没有任何可继续执行的 merge。
            if best_pair is None:
                break

            merged_pieces: list[bytes] = []
            index = 0

            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and pieces[index] == best_pair[0]
                    and pieces[index + 1] == best_pair[1]
                ):
                    merged_pieces.append(
                        pieces[index] + pieces[index + 1]
                    )
                    index += 2
                else:
                    merged_pieces.append(pieces[index])
                    index += 1

            pieces = merged_pieces

        result = tuple(pieces)
        self.bpe_cache[token_bytes] = result
        return result

    def _encode_ordinary_text(self, text: str) -> list[int]:
        token_ids: list[int] = []

        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            pretoken = match.group(0)
            pretoken_bytes = pretoken.encode("utf-8")

            for bpe_token in self._apply_bpe(pretoken_bytes):
                token_ids.append(self.token_to_id[bpe_token])

        return token_ids

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []

        if self.special_token_pattern is None:
            return self._encode_ordinary_text(text)

        token_ids: list[int] = []

        # 捕获组会让 split 的结果里保留 special token 本身。
        sections = self.special_token_pattern.split(text)

        for section in sections:
            if section == "":
                continue

            if section in self.special_token_to_id:
                token_ids.append(
                    self.special_token_to_id[section]
                )
            else:
                token_ids.extend(
                    self._encode_ordinary_text(section)
                )

        return token_ids

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return decoded_bytes.decode(
            "utf-8",
            errors="replace",
        )

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

def _merge_pair_in_word(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged_word: list[bytes] = []
    index = 0

    while index < len(word):
        if (
            index + 1 < len(word)
            and word[index] == pair[0]
            and word[index + 1] == pair[1]
        ):
            merged_word.append(word[index] + word[index + 1])
            index += 2
        else:
            merged_word.append(word[index])
            index += 1

    return tuple(merged_word)


def _get_adjacent_pair_counts(
    word: tuple[bytes, ...],
) -> Counter[tuple[bytes, bytes]]:
    return Counter(zip(word, word[1:]))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    unique_special_tokens = list(dict.fromkeys(special_tokens))

    minimum_vocab_size = 256 + len(unique_special_tokens)

    if vocab_size < minimum_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {minimum_vocab_size}"
        )

    # 初始词表：256 种单字节。
    vocab: dict[int, bytes] = {
        byte_value: bytes([byte_value])
        for byte_value in range(256)
    }

    # special token 单独加入词表，不参与 merge。
    for special_token in unique_special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    with open(input_path, encoding="utf-8") as input_file:
        text = input_file.read()

    # special token 用作语料边界，避免它与周围字符发生合并。
    if unique_special_tokens:
        sorted_special_tokens = sorted(
            unique_special_tokens,
            key=len,
            reverse=True,
        )

        special_pattern = regex.compile(
            "|".join(
                regex.escape(token)
                for token in sorted_special_tokens
            )
        )

        ordinary_sections = special_pattern.split(text)
    else:
        ordinary_sections = [text]

    pretoken_counts: Counter[bytes] = Counter()

    for section in ordinary_sections:
        for match in GPT2_PRETOKEN_PATTERN.finditer(section):
            pretoken_counts[match.group(0).encode("utf-8")] += 1

    # 每种不同的 pre-token 只保存一次，同时记录它在语料中的频次。
    words: list[tuple[bytes, ...]] = []
    word_frequencies: list[int] = []

    for pretoken, frequency in pretoken_counts.items():
        words.append(
            tuple(bytes([byte]) for byte in pretoken)
        )
        word_frequencies.append(frequency)

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    # 反向索引：
    # 某个 pair 出现在哪些不同的 pre-token 中。
    pair_to_word_ids: defaultdict[
        tuple[bytes, bytes],
        set[int],
    ] = defaultdict(set)

    for word_id, (word, frequency) in enumerate(
        zip(words, word_frequencies)
    ):
        local_pair_counts = _get_adjacent_pair_counts(word)

        for pair, local_count in local_pair_counts.items():
            pair_counts[pair] += local_count * frequency
            pair_to_word_ids[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []

    number_of_merges = vocab_size - len(vocab)

    for _ in range(number_of_merges):
        if not pair_counts:
            break

        # 优先选择：
        # 1. 频次最高的 pair
        # 2. 频次相同时，按 bytes tuple 的字典序选更大的 pair
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_word_ids = list(
            pair_to_word_ids.get(best_pair, set())
        )

        for word_id in affected_word_ids:
            old_word = words[word_id]
            frequency = word_frequencies[word_id]

            old_local_counts = _get_adjacent_pair_counts(old_word)

            new_word = _merge_pair_in_word(
                old_word,
                best_pair,
            )

            new_local_counts = _get_adjacent_pair_counts(new_word)

            old_pairs = set(old_local_counts)
            new_pairs = set(new_local_counts)

            # 先扣除这个 word 对旧 pair 频次的贡献。
            for pair, local_count in old_local_counts.items():
                pair_counts[pair] -= local_count * frequency

                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            # 再加入合并后对新 pair 频次的贡献。
            for pair, local_count in new_local_counts.items():
                pair_counts[pair] += local_count * frequency

            # 更新 pair 到 word 的反向索引。
            for removed_pair in old_pairs - new_pairs:
                pair_to_word_ids[removed_pair].discard(word_id)

                if not pair_to_word_ids[removed_pair]:
                    del pair_to_word_ids[removed_pair]

            for added_pair in new_pairs - old_pairs:
                pair_to_word_ids[added_pair].add(word_id)

            words[word_id] = new_word

    return vocab, merges