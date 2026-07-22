import heapq
import regex
from collections import Counter, defaultdict


GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class _ReversePair:
    """让 heapq 在频率相同时优先选择字典序更大的 pair。"""

    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def _merge_pair(tokens, pair, new_token):
    """将 token 序列中所有不重叠的指定 pair 合并。"""
    first, second = pair
    merged = []
    index = 0

    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == first
            and tokens[index + 1] == second
        ):
            merged.append(new_token)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1

    return merged


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    """
    使用增量 pair 统计训练 Byte-Level BPE。
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            "vocab_size must be at least 256 + len(special_tokens)"
        )

    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens cannot contain duplicates")

    if any(token == "" for token in special_tokens):
        raise ValueError("special_tokens cannot contain empty strings")

    byte_tokens = tuple(bytes([index]) for index in range(256))
    vocab = {index: byte_tokens[index] for index in range(256)}
    merges = []
    next_id = 256

    # 第一步：预分词并统计不同单词的频率
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    word_counts = defaultdict(int)
    compiled_pattern = regex.compile(GPT2_PATTERN)
    special_token_set = set(special_tokens)

    if special_tokens:
        # 较长的特殊 token 优先匹配
        escaped_tokens = [
            regex.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        ]
        split_pattern = regex.compile(
            f"({'|'.join(escaped_tokens)})"
        )
        pieces = split_pattern.split(text)
    else:
        pieces = [text]

    for piece in pieces:
        if not piece or piece in special_token_set:
            continue

        for match in compiled_pattern.finditer(piece):
            encoded = match.group().encode("utf-8")
            word = tuple(byte_tokens[byte] for byte in encoded)
            word_counts[word] += 1

    del text
    del pieces

    # 每个不同单词使用一个 ID，避免不断重建整个字典
    words = [list(word) for word in word_counts]
    frequencies = list(word_counts.values())
    del word_counts

    # pair_counts[pair]：pair 在全部语料中的加权出现次数
    # pair_to_words[pair]：包含该 pair 的单词 ID
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for word_id, (tokens, frequency) in enumerate(
        zip(words, frequencies)
    ):
        local_counts = Counter(zip(tokens, tokens[1:]))

        for pair, occurrences in local_counts.items():
            pair_counts[pair] += occurrences * frequency
            pair_to_words[pair].add(word_id)

    # 使用最大堆维护最高频 pair
    heap = [
        (-count, _ReversePair(pair))
        for pair, count in pair_counts.items()
        if count > 0
    ]
    heapq.heapify(heap)

    num_merges = vocab_size - 256 - len(special_tokens)

    for step in range(num_merges):
        # 清除计数已经过期的堆元素
        while heap:
            negative_count, wrapped_pair = heapq.heappop(heap)
            best_pair = wrapped_pair.pair
            current_count = pair_counts.get(best_pair, 0)

            if current_count == -negative_count and current_count > 0:
                break
        else:
            break

        first, second = best_pair
        new_token = first + second

        vocab[next_id] = new_token
        next_id += 1
        merges.append(best_pair)

        # 只处理包含 best_pair 的单词
        affected_word_ids = tuple(
            pair_to_words.get(best_pair, ())
        )
        pair_deltas = defaultdict(int)

        for word_id in affected_word_ids:
            old_tokens = words[word_id]
            frequency = frequencies[word_id]

            old_local_counts = Counter(
                zip(old_tokens, old_tokens[1:])
            )

            new_tokens = _merge_pair(
                old_tokens,
                best_pair,
                new_token,
            )
            new_local_counts = Counter(
                zip(new_tokens, new_tokens[1:])
            )

            changed_pairs = (
                old_local_counts.keys()
                | new_local_counts.keys()
            )

            for pair in changed_pairs:
                old_occurrences = old_local_counts.get(pair, 0)
                new_occurrences = new_local_counts.get(pair, 0)

                pair_deltas[pair] += (
                    new_occurrences - old_occurrences
                ) * frequency

                # 更新 pair 到单词的倒排索引
                if old_occurrences and not new_occurrences:
                    word_ids = pair_to_words.get(pair)
                    if word_ids is not None:
                        word_ids.discard(word_id)
                        if not word_ids:
                            pair_to_words.pop(pair, None)

                elif new_occurrences and not old_occurrences:
                    pair_to_words[pair].add(word_id)

            words[word_id] = new_tokens

        # 一次性更新受影响 pair 的全局计数
        for pair, delta in pair_deltas.items():
            new_count = pair_counts.get(pair, 0) + delta

            if new_count <= 0:
                pair_counts.pop(pair, None)
            else:
                pair_counts[pair] = new_count
                heapq.heappush(
                    heap,
                    (-new_count, _ReversePair(pair)),
                )

        # 避免过期堆元素积累过多
        if len(heap) > max(1_000_000, 8 * len(pair_counts)):
            heap = [
                (-count, _ReversePair(pair))
                for pair, count in pair_counts.items()
                if count > 0
            ]
            heapq.heapify(heap)

        if (step + 1) % 1000 == 0:
            print(
                f"merge {step + 1:,}/{num_merges:,}, "
                f"pair frequency={current_count:,}"
            )

    # 保持你原来代码中的特殊 token ID 顺序
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    return vocab, merges