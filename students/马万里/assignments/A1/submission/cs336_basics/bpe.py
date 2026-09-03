import heapq
import regex
from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

# GPT-2 预分词正则
PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class BPEHeap:
    """用于堆排序的包装器，使频率相同时字典序更大的 pair 排在前面。"""
    # 正常情况下每个python对象都有一个__dict__字典，通过__slots__限制该类实例只能拥有pair这一个属性
    __slots__ = ("pair",)

    def __init__(self, pair: Tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: "BPEHeap") -> bool:
        return self.pair > other.pair


def train_bpe(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 去重
    special_tokens = list(dict.fromkeys(special_tokens))
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab_values = set(vocab.values())
    next_token_id = 256

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab_values:
            vocab[next_token_id] = token_bytes
            vocab_values.add(token_bytes)
            next_token_id += 1

    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size is too small.")

    text = Path(input_path).read_text(encoding="utf-8")

    # 按特殊 token 分割，并保留分隔符
    if special_tokens:
        ordered_special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = regex.compile(
            "(" + "|".join(regex.escape(tok) for tok in ordered_special_tokens) + ")"
        )
        segments = special_pattern.split(text)
        special_token_lookup = set(special_tokens)
    else:
        segments = (text,)
        special_token_lookup = set()

    # 预分词并统计每个唯一 pre-token 的频率
    byte_tokens = tuple(bytes([i]) for i in range(256))
    pre_token_counts: Counter[Tuple[bytes, ...]] = Counter()

    for part in segments:
        if not part or part in special_token_lookup:
            continue
        for match in PAT.finditer(part):
            pre_token_bytes = match.group(0).encode("utf-8")
            pre_token = tuple(map(byte_tokens.__getitem__, pre_token_bytes))
            if pre_token:
                pre_token_counts[pre_token] += 1
                
    words: List[Tuple[bytes, ...]] = list(pre_token_counts.keys())
    word_freqs: List[int] = [pre_token_counts[w] for w in words]

    # pair 统计和反向索引
    pair_counts: Counter[Tuple[bytes, bytes]] = Counter()
    pair_to_words: Dict[Tuple[bytes, bytes], set] = defaultdict(set)

    for word_id, word in enumerate(words):
        freq = word_freqs[word_id]
        # 保证之后添加的每个word都只被添加一次
        pairs_visited = set()
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pairs_visited.add(pair)
        for pair in pairs_visited:
            pair_to_words[pair].add(word_id)

    merges: List[Tuple[bytes, bytes]] = []

    # 初始化堆，元素为 (-count, BPEHeap(pair), pair)
    pair_heap = [
        (-count, BPEHeap(pair), pair)
        for pair, count in pair_counts.items()
        if count > 0
    ]
    heapq.heapify(pair_heap)

    # 在 word 内部非重叠地合并指定 pair
    def _merge_pair_in_word(
        word: Tuple[bytes, ...],
        pair: Tuple[bytes, bytes],
    ) -> Tuple[bytes, ...]:
        merged: List[bytes] = []
        i = 0
        while i < len(word):
            if (i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]):
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return tuple(merged)

    # BPE merge
    merge_count = 0
    while len(vocab) < vocab_size and pair_counts:
        best_pair = None

        # 惰性堆：跳过过时条目
        while pair_heap:
            neg_count, _, candidate_pair = heapq.heappop(pair_heap)
            candidate_count = -neg_count
            if pair_counts.get(candidate_pair, 0) == candidate_count:
                best_pair = candidate_pair
                break

        if best_pair is None or pair_counts.get(best_pair, 0) <= 0:
            break

        # 合并为新 token
        merged_token = best_pair[0] + best_pair[1]
        vocab[next_token_id] = merged_token
        next_token_id += 1
        merges.append(best_pair)
        merge_count += 1

        if merge_count % 1000 == 0:
            print(
                f"Merge {merge_count}: "
                f"vocab={len(vocab)}, "
                f"pairs={len(pair_counts)}"
            )

        # 更新受影响的 pre-token
        affected_word_ids = tuple(pair_to_words.get(best_pair, ()))
        changed_pairs = set()

        for word_id in affected_word_ids:
            old_word = words[word_id]
            freq = word_freqs[word_id]

            old_pairs = list(zip(old_word, old_word[1:]))
            changed_pairs.update(old_pairs)

            # 移除该 word 对旧 pair 统计的贡献
            for pair in old_pairs:
                pair_counts[pair] -= freq
            for pair in set(old_pairs):
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word_id)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

            # 在 word 内部合并 best_pair
            new_word = _merge_pair_in_word(old_word, best_pair)
            words[word_id] = new_word

            new_pairs = list(zip(new_word, new_word[1:]))
            changed_pairs.update(new_pairs)

            # 加入该 word 对新 pair 统计的贡献
            for pair in new_pairs:
                pair_counts[pair] += freq
            for pair in set(new_pairs):
                pair_to_words[pair].add(word_id)

        for pair in changed_pairs:
            if pair_counts.get(pair, 0) <= 0:
                pair_counts.pop(pair, None)
            else:
                heapq.heappush(
                    pair_heap,
                    (-pair_counts[pair], BPEHeap(pair), pair)
                )

    print(f"Total merges: {merge_count}")

    return vocab, merges