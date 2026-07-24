from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import regex as re


GPT2_PRETOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练字节级 BPE tokenizer，并返回词表和按生成顺序排列的 merges。

    该函数保留作业 Part 2 的训练语义，并使用增量统计减少重复扫描：
    1. 读取输入语料。
    2. 按 special token 切分文本，确保 merge 不会跨过 special token 的边界。
    3. 使用 GPT-2 正则表达式对普通文本片段做预分词。
    4. 将每个 pretoken 初始化为字节 token 序列。
    5. 初始化全局 pair 计数及 pair 到 pretoken 的倒排索引。
    6. 每轮只更新包含最佳 pair 的 pretoken，避免重复全量扫描。
    7. 基于初始字节 token、special token 和学习到的 merge 构建最终词表。
    """
    normalized_special_tokens = validate_train_bpe_inputs(input_path, vocab_size, special_tokens)
    text = read_training_text(input_path)
    text_segments = split_on_special_tokens(text, normalized_special_tokens)
    word_counts = count_pretokens(text_segments)
    token_counts = initialize_byte_tokens(word_counts)
    vocab = initialize_vocab(normalized_special_tokens)
    merges = train_bpe_from_token_counts(token_counts, vocab, vocab_size)

    return vocab, merges


def validate_train_bpe_inputs(
    input_path: str,
    vocab_size: int,
    special_tokens: Sequence[str],
) -> list[str]:
    """校验公开 API 的输入，并返回去重后的 special tokens。

    special tokens 按首次出现顺序保留，保证词表 ID 分配结果确定。
    """
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"BPE 训练输入文件不存在: {input_path}")
    if vocab_size <= 0:
        raise ValueError("vocab_size 必须是正整数")
    if not all(isinstance(token, str) for token in special_tokens):
        raise TypeError("special_tokens 只能包含字符串")

    deduplicated_special_tokens = list(dict.fromkeys(special_tokens))
    minimum_vocab_size = 256 + len(deduplicated_special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(
            "vocab_size 至少需要等于 256 加上唯一 special token 的数量 "
            f"({minimum_vocab_size})"
        )

    return deduplicated_special_tokens


def read_training_text(input_path: str) -> str:
    """读取 UTF-8 编码的训练数据。"""
    return Path(input_path).read_text(encoding="utf-8")


def split_on_special_tokens(text: str, special_tokens: Sequence[str]) -> list[str]:
    """按 special token 将文本切分为普通片段，并丢弃 special token 片段。

    special tokens 是硬边界：它们不能参与 merge 统计，merge 也不能跨过它们的范围。
    """
    if not special_tokens:
        return [text]

    escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    special_token_pattern = "|".join(escaped_tokens)
    return [segment for segment in re.split(special_token_pattern, text) if segment]


def count_pretokens(text_segments: Iterable[str]) -> Counter[bytes]:
    """对文本片段做预分词，并按 UTF-8 bytes 统计每个 pretoken。"""
    pretoken_counts: Counter[bytes] = Counter()
    pattern = re.compile(GPT2_PRETOKENIZATION_PATTERN)

    for segment in text_segments:
        for match in pattern.finditer(segment):
            pretoken_counts[match.group(0).encode("utf-8")] += 1

    return pretoken_counts


def initialize_byte_tokens(word_counts: Mapping[bytes, int]) -> Counter[Token]:
    """将每个已计数的 pretoken 表示为单字节 BPE token 组成的元组。"""
    token_counts: Counter[Token] = Counter()
    for word, count in word_counts.items():
        token_counts[tuple(bytes([byte]) for byte in word)] += count
    return token_counts


def initialize_vocab(special_tokens: Sequence[str]) -> dict[int, bytes]:
    """创建初始字节级词表，并追加 special tokens。"""
    vocab: dict[int, bytes] = {byte: bytes([byte]) for byte in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab


def train_bpe_from_token_counts(
    token_counts: Mapping[Token, int],
    vocab: dict[int, bytes],
    vocab_size: int,
) -> list[Pair]:
    """使用增量 pair 统计训练 BPE，并原地扩展 ``vocab``。

    ``pair_to_word_ids`` 记录每种 pair 出现在哪些唯一 pretoken 中。每轮 merge
    只重算这些受影响的 pretoken，再用局部计数差更新全局统计。最佳 pair 仍通过
    全局计数和原始 bytes 字典序选择，因此行为与朴素实现一致。
    """
    word_tokens: list[Token] = []
    word_weights: list[int] = []
    word_pair_counts: list[dict[Pair, int]] = []
    pair_counts: dict[Pair, int] = {}
    pair_to_word_ids: dict[Pair, set[int]] = {}

    for token, weight in token_counts.items():
        if weight <= 0:
            continue

        word_id = len(word_tokens)
        local_pair_counts = count_token_pairs(token)
        word_tokens.append(token)
        word_weights.append(weight)
        word_pair_counts.append(local_pair_counts)

        for pair, occurrences in local_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * weight
            pair_to_word_ids.setdefault(pair, set()).add(word_id)

    merges: list[Pair] = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = select_best_pair(pair_counts)
        merged_token = best_pair[0] + best_pair[1]
        affected_word_ids = tuple(pair_to_word_ids.get(best_pair, ()))

        for word_id in affected_word_ids:
            old_pair_counts = word_pair_counts[word_id]
            merged_word, new_pair_counts = merge_pair_and_count_pairs(
                word_tokens[word_id],
                best_pair,
                merged_token,
            )
            word_tokens[word_id] = merged_word
            word_pair_counts[word_id] = new_pair_counts
            update_pair_statistics(
                old_pair_counts=old_pair_counts,
                new_pair_counts=new_pair_counts,
                word_id=word_id,
                word_weight=word_weights[word_id],
                pair_counts=pair_counts,
                pair_to_word_ids=pair_to_word_ids,
            )

        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

    return merges


def count_token_pairs(token: Sequence[bytes]) -> dict[Pair, int]:
    """统计单个 token 序列内的相邻 pair，保留重叠出现次数。"""
    pair_counts: dict[Pair, int] = {}
    for index in range(len(token) - 1):
        pair = (token[index], token[index + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def merge_pair_and_count_pairs(
    token: Sequence[bytes],
    pair_to_merge: Pair,
    merged_token: bytes,
) -> tuple[Token, dict[Pair, int]]:
    """从左到右合并非重叠 pair，并在同一次扫描中统计新 pair。"""
    merged_parts: list[bytes] = []
    pair_counts: dict[Pair, int] = {}
    left_token, right_token = pair_to_merge
    index = 0
    token_length = len(token)

    while index < token_length:
        if (
            index < token_length - 1
            and token[index] == left_token
            and token[index + 1] == right_token
        ):
            current_token = merged_token
            index += 2
        else:
            current_token = token[index]
            index += 1

        if merged_parts:
            pair = (merged_parts[-1], current_token)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        merged_parts.append(current_token)

    return tuple(merged_parts), pair_counts


def update_pair_statistics(
    old_pair_counts: Mapping[Pair, int],
    new_pair_counts: Mapping[Pair, int],
    word_id: int,
    word_weight: int,
    pair_counts: dict[Pair, int],
    pair_to_word_ids: dict[Pair, set[int]],
) -> None:
    """按单个 pretoken merge 前后的局部计数差更新全局统计和倒排索引。"""
    for pair in old_pair_counts.keys() | new_pair_counts.keys():
        old_occurrences = old_pair_counts.get(pair, 0)
        new_occurrences = new_pair_counts.get(pair, 0)
        if old_occurrences == new_occurrences:
            continue

        updated_count = pair_counts.get(pair, 0) + (
            new_occurrences - old_occurrences
        ) * word_weight
        if updated_count > 0:
            pair_counts[pair] = updated_count
        else:
            pair_counts.pop(pair, None)

        indexed_word_ids = pair_to_word_ids.get(pair)
        if new_occurrences > 0:
            if indexed_word_ids is None:
                pair_to_word_ids[pair] = {word_id}
            else:
                indexed_word_ids.add(word_id)
        elif indexed_word_ids is not None:
            indexed_word_ids.discard(word_id)
            if not indexed_word_ids:
                pair_to_word_ids.pop(pair, None)


def count_adjacent_pairs(token_counts: Counter[Token]) -> dict[Pair, int]:
    """统计相邻 BPE token pair，并按 pretoken 频次加权。"""
    pair_counts: dict[Pair, int] = {}
    for token, count in token_counts.items():
        for index in range(len(token) - 1):
            pair = (token[index], token[index + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


def select_best_pair(pair_counts: Mapping[Pair, int]) -> Pair:
    """选择下一组要 merge 的 token pair：先比较频次，再按字节字典序打破平局。"""
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]


def apply_merge(token_counts: Counter[Token], pair_to_merge: Pair) -> Counter[Token]:
    """将选中的 merge 应用到每个已 tokenized 的 pretoken。"""
    merged_token_counts: dict[Token, int] = {}
    left_token, right_token = pair_to_merge
    merged_token = left_token + right_token

    for token, count in token_counts.items():
        merged_parts: list[bytes] | None = None
        index = 0
        token_length = len(token)

        while index < token_length:
            if (
                index < token_length - 1
                and token[index] == left_token
                and token[index + 1] == right_token
            ):
                if merged_parts is None:
                    merged_parts = list(token[:index])
                merged_parts.append(merged_token)
                index += 2
            else:
                if merged_parts is not None:
                    merged_parts.append(token[index])
                index += 1

        if merged_parts is None:
            merged_token_counts[token] = merged_token_counts.get(token, 0) + count
        else:
            merged_token_tuple = tuple(merged_parts)
            merged_token_counts[merged_token_tuple] = merged_token_counts.get(merged_token_tuple, 0) + count

    return Counter(merged_token_counts)
