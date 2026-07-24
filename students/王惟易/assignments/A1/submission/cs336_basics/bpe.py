import os
from collections import Counter
from dataclasses import dataclass
import heapq
from typing import BinaryIO

import regex as re

type Pretoken = tuple[bytes, ...]
type Pair = tuple[bytes, bytes]

BYTE_TOKENS = tuple(bytes([value]) for value in range(256))

@dataclass(frozen=True, slots=True)
class _PairCandidate:
    count: int
    pair: Pair

    def __lt__(self, other: "_PairCandidate") -> bool:
        # heapq 是最小堆
        # 让 rank 更大的候选更小
        if self.count != other.count:
            return self.count > other.count # larger count == "smaller" in heap order
        return self.pair > other.pair

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_pretokens_from_file(
        input_path: str | os.PathLike[str],
        special_tokens: list[str],
        desired_num_chunks: int = 32,
) -> dict[tuple[bytes, ...], int]:
    total_counts: Counter[tuple[bytes, ...]] = Counter()

    if not special_tokens:
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        total_counts.update(count_pretokens(text, special_tokens))
        return total_counts

    split_token_bytes = special_tokens[0].encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, split_token_bytes)

        for start, end in zip(boundaries, boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8")
            chunk_counts = count_pretokens(chunk_text, special_tokens)
            total_counts.update(chunk_counts)

            del chunk_text, chunk_counts, chunk_bytes

    return total_counts


def initialize_pair_heap(
        pair_counts: Counter[Pair],
) -> list[_PairCandidate]:
    heap = [_PairCandidate(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    return heap


def pop_best_pair(
        heap: list[_PairCandidate],
        pair_counts: Counter[Pair],
) -> Pair:
    while heap:
        candidate = heapq.heappop(heap)                 # 弹出堆顶
        current_count = pair_counts.get(candidate.pair) # 从 pair_counts 查询真实计数（heap可能过期）

        if candidate.count == current_count:
            return candidate.pair

        # 否则丢弃并继续
        # 如果一个最终计数没变得 pair 再次被 heappush，则 heap 中会有两个完全相同且都有效的候选。但是无论弹出哪一个，得到的都是同一个正确 pair；它被 merge 后，剩余的副本会因计数不匹配而成为 stale entry，当检查到时就会被删除。

    raise ValueError("no pair candidates")

def initialize_pair_index(
        pretoken_counts: dict[Pretoken, int],
) -> tuple[
    list[Pretoken],
    list[int],
    Counter[Pair],
    dict[Pair, set[int]],
]:
    pretokens = []
    frequencies = []
    pair_counts = Counter()
    pair_to_pretoken_ids = {}

    for pretoken_id, (pretoken, frequency) in enumerate(pretoken_counts.items()):
        pretokens.append(pretoken)
        frequencies.append(frequency)

        for pair in zip(pretoken, pretoken[1:]):
            pair_counts[pair] += frequency
            pair_to_pretoken_ids.setdefault(pair, set()).add(pretoken_id)

    return pretokens, frequencies, pair_counts, pair_to_pretoken_ids


def initialize_vocab(
        special_tokens: list[str]
) -> dict[int, bytes]:
    """
    Initialize a vocab with special tokens.
    """
    vocab = {}
    for i in range(256):
        vocab[i] = BYTE_TOKENS[i]
    for i, token in enumerate(special_tokens):
        token_bytes = token.encode("utf-8")
        vocab[256 + i] = token_bytes
    return vocab


def count_pretokens(
        text: str,
        special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    Count the number of pre-tokens in a text.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    delimiter = "|".join(re.escape(token) for token in special_tokens)
    segments = re.splititer(delimiter, text) if special_tokens else (text,)

    # 使用 collections.Counter 来统计每个 pre-token 的数量
    pretoken_counter = Counter()

    # 三层结构：先按分割符分割文本，然后对每个 segment 使用正则表达式匹配 pre-token，最后把每个 pre-token 转换为 bytes 并统计数量
    for segment in segments:
        for match in re.finditer(PAT, segment):
            raw = match.group().encode("utf-8")
            pretoken = tuple(BYTE_TOKENS[b] for b in raw)
            pretoken_counter[pretoken] += 1

    return pretoken_counter


def count_pairs(
        pretoken_counts: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    """
    Count the number of pairs of pre-tokens.
    """
    pair_counter = Counter()
    for pretoken, count in pretoken_counts.items():
        if len(pretoken) < 2:
            continue
        for left, right in zip(pretoken, pretoken[1:]):
            pair = (left, right)
            pair_counter[pair] += count

    return pair_counter


def _choose_best_pair(
        pair_counts: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes]:
    """
    Choose the best pair of pre-tokens to merge.
    first key: frequency
    second key: lexicographical order of the pair
    """
    """
    在所有 pair 中找 best pair 成为新的瓶颈。最新实现(pop_best_pair)使用最小堆来找 best pair
    """
    # 未来由训练循环决定没有 pair 时是否终止，因此不加 pair_counts 为空的检查
    return max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]


def merge_pretoken(
        pretoken: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged_pretoken = []
    left = 0
    while left < len(pretoken):
        right = left + 1
        if right < len(pretoken) and (pretoken[left], pretoken[right]) == pair:
            merged_pretoken.append(pretoken[left]+pretoken[right])
            left += 2
        else:
            merged_pretoken.append(pretoken[left])
            left += 1

    return tuple(merged_pretoken)


def _apply_merge_naive(
        pretoken_counts: dict[Pretoken, int],
        pair: Pair
) -> dict[Pretoken, int]:
    """Old, naive implementation that rebuilds every pre-token"""
    merged_pretoken_counts = Counter()
    for pretoken, count in pretoken_counts.items():
        merged_pretoken = merge_pretoken(pretoken, pair)
        if merged_pretoken_counts.get(merged_pretoken) is not None:
            merged_pretoken_counts[merged_pretoken] += count
        else:
            merged_pretoken_counts[merged_pretoken] = count

    return merged_pretoken_counts


def apply_merge(
        pretoken_counts: dict[tuple[bytes, ...], int],
        pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """
    Merge the selected pair while reusing unaffected pre-token tuples
    """
    merged_pretoken_counts = Counter()
    for pretoken, count in pretoken_counts.items():
        # 优化后的写法，先检查是否包含 pair，如果包含才重新构造 pretoken
        if pair in zip(pretoken, pretoken[1:]):
            merged_pretoken = merge_pretoken(pretoken, pair)
        else:
            merged_pretoken = pretoken

        merged_pretoken_counts[merged_pretoken] += count

    return merged_pretoken_counts


def apply_indexed_merge(
        pretokens: list[Pretoken],
        frequencies: list[int],
        pair_counts: Counter[Pair],
        pair_to_pretoken_ids: dict[Pair, set[int]],
        pair: Pair,
) -> set[Pair]:
    changed_pairs: set[Pair] = set()
    affected_ids = tuple(pair_to_pretoken_ids[pair])

    for pretoken_id in affected_ids:
        old_pretoken = pretokens[pretoken_id]
        frequency = frequencies[pretoken_id]
        old_pairs = tuple(zip(old_pretoken, old_pretoken[1:]))
        changed_pairs.update(old_pairs)
        for old_pair in old_pairs:
            pair_counts[old_pair] -= frequency
            if pair_counts[old_pair] == 0:
                del pair_counts[old_pair]
        for old_pair in set(old_pairs):
            ids = pair_to_pretoken_ids[old_pair]
            ids.remove(pretoken_id)
            if not ids:
                del pair_to_pretoken_ids[old_pair]
        new_pretoken = merge_pretoken(old_pretoken, pair)
        pretokens[pretoken_id] = new_pretoken
        new_pairs = tuple(zip(new_pretoken, new_pretoken[1:]))
        changed_pairs.update(new_pairs)
        for new_pair in new_pairs:
            pair_counts[new_pair] += frequency
            pair_to_pretoken_ids.setdefault(new_pair, set()).add(pretoken_id)

    return changed_pairs


def _train_bpe_naive(
        input_path: str | os.PathLike[str],
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    串行版本。
    1. 以 utf-8 读取文件
    2. initialize_vocab
    3. count_pretokens
    4. 初始化空的 merges
    5. 在 len(vocab) < vocab_size 时循环：
        - 重新 count_pairs
        - 如果没有 pairs，提前结束
        - 调用 _choose_best_pair
        - 将 pair[0] + pair[1] 以新的 ID 加入词表
        - 将原始 pair 追加到 merges
        - 调用 apply_merge 更新 pretoken 技数
    6. 返回 (vocab, merges)
    """
    # read file
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    vocab = initialize_vocab(special_tokens)
    pretoken_counts = count_pretokens(text, special_tokens)
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(pretoken_counts)
        if not pair_counts:
            return vocab, merges
        best_pair = _choose_best_pair(pair_counts)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        pretoken_counts = apply_merge(pretoken_counts, best_pair)

    return vocab, merges


def train_bpe(
        input_path: str | os.PathLike[str],
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """优化版本，利用pair_index，相比 _train_bpe_naive 实现 ~8.6 倍加速！
    使用 heap 来维护 best pair 之后进一步有 ~2.4 倍加速"""
    vocab = initialize_vocab(special_tokens)
    pretoken_counts = count_pretokens_from_file(input_path, special_tokens)
    merges = []
    pretokens, frequencies, pair_counts, pair_index = initialize_pair_index(
        pretoken_counts,
    )
    pair_heap = initialize_pair_heap(pair_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            return vocab, merges
        best_pair = pop_best_pair(pair_heap, pair_counts)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        changed_pairs = apply_indexed_merge(pretokens, frequencies, pair_counts, pair_index, best_pair)

        for changed_pair in changed_pairs:
            current_count = pair_counts.get(changed_pair)
            if current_count is not None:
                heapq.heappush(
                    pair_heap,
                    _PairCandidate(current_count, changed_pair)
                )

    return vocab, merges


if __name__ == "__main__":
    special = "<|endoftext|>"
    vocab = initialize_vocab([special])

    print(len(vocab))
    print(vocab[0], vocab[255], vocab[256])

    counts = count_pretokens(
    f"low{special}lower{special}lower",
    [special],
    )

    for pretoken, count in counts.items():
        print(pretoken, count)

    pretoken_counts: dict[tuple[bytes,...],int] = {
        (b"a", b"a", b"a"): 2,
    }

    pair_counts = count_pairs(pretoken_counts)
    print(pair_counts)

    pair_counts = {
        (b"A", b"B"): 9,
        (b"A", b"C"): 9,
        (b"B", b"ZZ"): 9,
        (b"BA", b"A"): 9
    }

    print(_choose_best_pair(pair_counts))

    print(count_pairs({
        (b"a",): 10,
        (b"b",): 10
    }))

    print(merge_pretoken(
        (b"l",b"ow",b"a",b"r"),
        (b"l",b"ow")
    ))

    counts: dict[tuple[bytes, ...], int] = {
        (b"a", b"b", b"c"): 2,
        (b"ab", b"c"): 4,
    }
    merged = apply_merge(counts, (b"a", b"b"))
    print(merged)

    counts: dict[Pretoken, int] = {
        (b"l", b"o", b"w"): 5,
        (b"l", b"o", b"w", b"e", b"r"): 2,
        (b"w", b"i", b"d", b"e", b"s", b"t"): 3,
        (b"n", b"e", b"w", b"e", b"s", b"t"): 6,
    }

    pretokens, frequencies, pair_counts, pair_index = initialize_pair_index(counts)

    apply_indexed_merge(
        pretokens,
        frequencies,
        pair_counts,
        pair_index,
        (b"s", b"t"),
    )

    print(pretokens[2])
    print(pretokens[3])
    print(frequencies)
    print(pair_counts[(b"e", b"st")])
    print(pair_index[(b"e", b"st")])
