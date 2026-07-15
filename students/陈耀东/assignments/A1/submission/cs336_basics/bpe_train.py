from __future__ import annotations

import os
import heapq
import mmap
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import regex

# GPT-2 使用的预分词正则表达式。
# 说明：
# - r"""...""" 是 Python 的 raw string，反斜杠不会被 Python 提前转义。
# - 这个 pattern 需要第三方 regex 包，而不是 Python 内置 re 包，
#   因为它用到了 \p{L}、\p{N} 这种 Unicode 字符类别。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# 下面三个是“类型别名”，不是新类型，只是为了让注释更好读。
# bytes：Python 的字节串类型，例如 b"a"、b"\xe7\x89\x9b"。
# tuple[...]：元组，长度和内容通常固定；这里用来表示一个 pre-token 内的 token 序列。
# list[...]：列表，顺序可变；这里 merges 需要按创建顺序保存，所以用 list。
# dict[K, V]：字典，从 key 映射到 value；本任务里 vocab 是 token_id -> token_bytes。
# Counter：一种特殊字典，适合做“某个东西出现了多少次”的计数。
Token = bytes
TokenPair = tuple[Token, Token]
Pretoken = tuple[Token, ...]

_BYTE_TOKENS = tuple(bytes([value]) for value in range(256))
_PARALLEL_MIN_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_PROCESSES = 16


@dataclass(slots=True)
class _PairHeapEntry:
    """让 heapq 按“频率更高、pair 字典序更大”的顺序弹出。"""

    count: int
    pair: TokenPair

    def __lt__(self, other: _PairHeapEntry) -> bool:
        return (self.count, self.pair) > (other.count, other.pair)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练一个 byte-level BPE tokenizer。

    这是学习脚手架，不是完整答案。建议你按 helper function 一个个补。

    Python 语法小注释：
    - def：定义函数。
    - input_path: str | os.PathLike：参数类型提示，表示可以传字符串路径或路径对象。
    - -> tuple[...]：返回值类型提示，表示函数返回一个二元组。
    - Path(input_path)：把普通字符串路径转换成更好用的 Path 对象。

    整体思路：
    1. 建立初始词表：256 个单 byte token + special tokens。
    2. 读取语料，并按 special token 分割；special token 是硬边界。
    3. 对普通文本片段做 GPT-2 regex pre-tokenization。
    4. 把每个 pre-token 表示成 tuple[bytes, ...]。
    5. 反复选择最高频相邻 pair，合并，更新 vocab 和 merges。

    测试命令：
        ..\\scripts\\cs336-run.cmd uv run pytest tests/test_train_bpe.py -q
    """
    input_path = Path(input_path)

    # vocab 是 dict[int, bytes]：
    # - key 是 token id，例如 0、1、2、...
    # - value 是这个 token 对应的 bytes，例如 b"a" 或 b"the"
    vocab = build_initial_vocab(special_tokens)

    # pretoken_counts 是 dict[Pretoken, int]：
    # - key 是一个 pre-token 的 byte-token 序列，例如 (b"l", b"o", b"w")
    # - value 是它在语料中出现的次数
    pretoken_counts = count_pretokens(
        input_path,
        special_tokens,
        num_processes=num_processes,
    )

    # merges 是 list[tuple[bytes, bytes]]：
    # - 每个元素是一次 merge，例如 (b"t", b"h")
    # - 顺序很重要，必须按训练产生的顺序保存
    merges: list[tuple[bytes, bytes]] = []

    # vocab_size 包含：256 个初始 byte token + special tokens + merge 产生的新 token。
    num_merges_to_learn = vocab_size - len(vocab)
    if num_merges_to_learn < 0:
        raise ValueError("vocab_size 小于 256 + special token 数量，无法训练")

    # 激进优化版本：
    # - 不再每轮从零扫描所有 pre-token 重新统计 pair。
    # - pair_counts 保存当前每个 pair 的总频率。
    # - pair_to_pretokens 保存“某个 pair 出现在哪些 pre-token 中”。
    # - 每次 merge 后，只更新真正包含 best_pair 的 pre-token。
    pair_counts, pair_to_pretokens = build_pair_indexes(pretoken_counts)
    pair_heap = build_pair_heap(pair_counts)

    # range(n) 会产生 0, 1, ..., n-1；这里表示最多学习 n 次 merge。
    for _ in range(num_merges_to_learn):
        best_pair = pop_best_pair(pair_heap, pair_counts)
        if best_pair is None:
            break
        merges.append(best_pair)

        # 新 token 的 bytes 就是左右两个旧 token 的 bytes 拼接。
        # 例如 b"t" + b"h" 得到 b"th"。
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        changed_pairs = merge_pair_incremental_with_changes(
            pretoken_counts=pretoken_counts,
            pair_counts=pair_counts,
            pair_to_pretokens=pair_to_pretokens,
            pair_to_merge=best_pair,
        )

        for pair in changed_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(pair_heap, _PairHeapEntry(count=count, pair=pair))

        # 惰性删除会保留旧 heap entry；过大时偶尔重建一次，避免内存持续膨胀。
        if len(pair_heap) > max(100_000, 4 * len(pair_counts)):
            pair_heap = build_pair_heap(pair_counts)

    return vocab, merges


def build_initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """创建初始词表：token_id -> token_bytes。

    你要补的逻辑：
    - 先加入 256 个单 byte token。
    - 再把 special_tokens 逐个用 UTF-8 编码后追加到词表末尾。

    Python 语法/数据结构提示：
    - {} 可以创建空字典。
    - bytes([i]) 可以把 0..255 的整数转换成单 byte 的 bytes。
      例如 bytes([97]) == b"a"。
    - "abc".encode("utf-8") 会把 str 转成 bytes。
    - vocab[len(vocab)] = value 表示用当前词表长度作为下一个 token id。

    自查例子：
    - token id 0 应该对应 b"\\x00"。
    - token id 97 应该对应 b"a"。
    - special token 应该出现在 id 256 及之后。
    """
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab


def count_pretokens(
    input_path: Path,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> dict[Pretoken, int]:
    """读取语料，并统计 byte-level pre-token 出现次数。

    这一步的目标：
    - 输入：一个文本文件路径 input_path，以及 special_tokens 列表。
    - 输出：一个字典/Counter，记录每个 pre-token 的 byte-token 序列出现了多少次。
    - 注意：special token 只作为“切分边界”，不应该被统计进普通 pre-token。

    最小例子：
    - 假设文本是："low low<|endoftext|>new"
    - special_tokens 是 ["<|endoftext|>"]
    - 应先切成两个普通文本片段："low low" 和 "new"
    - "<|endoftext|>" 本身不参与 pair frequency，也不变成普通 pre-token。

    详细步骤 1：读取文本
    - Path 对象可以用 input_path.read_text(encoding="utf-8") 读取完整文本。
    - 读出来的 text 是 Python str，不是 bytes。
    - str 表示 Unicode 字符序列；后面要再 encode 成 UTF-8 bytes。

    详细步骤 2：准备 Counter
    - Counter() 像一个“默认值为 0 的计数字典”。
    - 例如 counter[x] += 1 表示 x 的出现次数加 1。
    - 本函数最后可以直接 return counter；Counter 也是 dict 的一种近亲。

    详细步骤 3：按 special token 分割文本
    - 如果没有 special_tokens，可以把整个 text 当作一个普通片段。
    - 如果有 special_tokens，需要把它们作为分隔符。
    - 注意 special token 里可能有 |、<、> 这种正则特殊字符，所以拼正则时要 escape。
    - 分割后的每个普通片段 segment 才进入 GPT-2 regex pre-tokenization。
    - 空 segment 可以跳过，例如连续 special token 之间可能切出空字符串。

    详细步骤 4：对每个普通片段做 regex pre-tokenization
    - 这里应使用第三方 regex 包：import regex
    - 使用 regex.finditer(PAT, segment) 可以依次得到每个 match。
    - match.group() 可以取出这次匹配到的字符串 pretoken_text。
    - pretoken_text 仍然是 str，还不是 bytes。

    详细步骤 5：把 str pre-token 变成 UTF-8 bytes
    - pretoken_text.encode("utf-8") 会得到 bytes。
    - 例子："牛".encode("utf-8") 得到 3 个 byte。
    - BPE 训练是在 byte-level 上做，所以必须转成 bytes。

    详细步骤 6：把一个 bytes 对象拆成“单 byte 的 bytes 元组”
    - 遍历 bytes 对象时，拿到的是 int，不是长度为 1 的 bytes。
    - 例如 list(b"abc") 得到 [97, 98, 99]。
    - bytes([97]) 才会得到 b"a"。
    - 所以 b"abc" 应该变成 (b"a", b"b", b"c")。
    - 这个 tuple 就是本任务里的 Pretoken。

    详细步骤 7：计数
    - counter[pretoken] += 1 表示这个 pre-token 又出现了一次。
    - 如果相同 pre-token 出现很多次，后续统计 pair frequency 时可以直接乘以 count。

    Python 语法/数据结构提示：
    - for x in something: 表示循环遍历。
    - if not segment: 可以判断字符串是否为空。
    - tuple(...) 会创建元组，元组可以作为 dict/Counter 的 key。
    - bytes([b]) 可以把一个 0..255 的整数 b 转成单 byte 的 bytes 对象。
    - list comprehension / generator expression 是 Python 常见写法，但你也可以先用普通 for 循环写清楚。

    形状例子：
    - 文本 pre-token: "low"
    - UTF-8 bytes: b"low"
    - tuple 表示: (b"l", b"o", b"w")
    """
    input_path = Path(input_path)
    worker_count = resolve_num_processes(input_path, special_tokens, num_processes)
    special_tokens_tuple = tuple(special_tokens)

    # 多取几倍于 worker 数量的块，使不同长度故事的负载更均匀。
    desired_chunks = max(1, worker_count * 4)
    boundaries = find_chunk_boundaries(
        input_path,
        desired_num_chunks=desired_chunks,
        split_special_tokens=tuple(token.encode("utf-8") for token in special_tokens),
    )
    ranges = list(zip(boundaries[:-1], boundaries[1:]))

    byte_counts: Counter[bytes] = Counter()
    tasks = [(str(input_path), start, end, special_tokens_tuple) for start, end in ranges if end > start]

    if worker_count == 1 or len(tasks) <= 1:
        for task in tasks:
            byte_counts.update(count_pretokens_in_range(task))
    else:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(tasks))) as executor:
            for partial_counts in executor.map(count_pretokens_in_range, tasks, chunksize=1):
                byte_counts.update(partial_counts)

    return {
        tuple(_BYTE_TOKENS[value] for value in pretoken_bytes): count for pretoken_bytes, count in byte_counts.items()
    }


def resolve_num_processes(
    input_path: Path,
    special_tokens: list[str],
    requested: int | None,
) -> int:
    """根据文件大小和 special token 条件选择预分词进程数。"""
    explicitly_requested = requested is not None
    if requested is None:
        env_value = os.environ.get("CS336_BPE_NUM_PROCESSES")
        explicitly_requested = env_value is not None
        requested = int(env_value) if env_value else min(os.cpu_count() or 1, _DEFAULT_MAX_PROCESSES)
    if requested < 1:
        raise ValueError("num_processes 必须至少为 1")
    if (not explicitly_requested and input_path.stat().st_size < _PARALLEL_MIN_BYTES) or not special_tokens:
        return 1
    if any(token == "" for token in special_tokens):
        raise ValueError("special token 不能为空字符串")
    return requested


def find_chunk_boundaries(
    input_path: Path,
    desired_num_chunks: int,
    split_special_tokens: tuple[bytes, ...],
) -> list[int]:
    """把大文件边界移动到下一个 special token，保证各块可独立计数。"""
    file_size = input_path.stat().st_size
    if file_size == 0:
        return [0]
    if desired_num_chunks <= 1 or not split_special_tokens:
        return [0, file_size]

    boundaries = [0]
    with input_path.open("rb") as input_file:
        with mmap.mmap(input_file.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            for index in range(1, desired_num_chunks):
                target = file_size * index // desired_num_chunks
                candidates = [mapped.find(token, target) for token in split_special_tokens]
                candidates = [position for position in candidates if position >= 0]
                boundaries.append(min(candidates) if candidates else file_size)
    boundaries.append(file_size)
    return sorted(set(boundaries))


def count_pretokens_in_range(
    task: tuple[str, int, int, tuple[str, ...]],
) -> Counter[bytes]:
    """读取一个安全文件块，返回以原始 UTF-8 bytes 为 key 的计数。"""
    input_path, start, end, special_tokens = task
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        text = input_file.read(end - start).decode("utf-8")
    # 与 Path.read_text 的 universal-newline 行为保持一致，保证跨平台 merge 结果相同。
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if len(special_tokens) == 1:
        segments = text.split(special_tokens[0])
    elif special_tokens:
        split_pattern = "|".join(regex.escape(token) for token in special_tokens)
        segments = regex.split(split_pattern, text)
    else:
        segments = [text]

    counts: Counter[bytes] = Counter()
    for segment in segments:
        for match in regex.finditer(PAT, segment):
            counts[match.group().encode("utf-8")] += 1
    return counts


def count_pair_frequencies(
    pretoken_counts: dict[Pretoken, int],
) -> Counter[TokenPair]:
    """统计所有 pre-token 内部的相邻 token pair 频率。

    重要规则：
    - pair 不跨 pre-token 边界。
    - pre-token 出现 count 次，其中的 pair 也贡献 count 次。
    - 长度为 0 或 1 的 pre-token 没有相邻 pair。

    Python 语法/数据结构提示：
    - dict.items() 会同时遍历 key 和 value。
      例如 for pretoken, count in pretoken_counts.items():
    - len(pretoken) 是序列长度。
    - pretoken[i] 是第 i 个元素。
    - (pretoken[i], pretoken[i + 1]) 可以组成一个二元组 pair。
    """
    pair_counts: Counter[TokenPair] = Counter()

    # 遍历每个 pre-token 和它的出现次数
    for pretoken, count in pretoken_counts.items():
        # 长度小于 2 的没有相邻 pair，跳过
        if len(pretoken) < 2:
            continue

        # 遍历相邻的每一对
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count

    return pair_counts


def count_pairs_in_one_pretoken(pretoken: Pretoken, count: int) -> Counter[TokenPair]:
    """统计单个 pre-token 内部 pair 频率。

    参数解释：
    - pretoken: 一个 pre-token，例如 (b"l", b"o", b"w")
    - count: 这个 pre-token 在语料中出现了几次

    返回：
    - Counter[TokenPair]，表示这个 pre-token 对全局 pair_counts 的贡献。

    为什么需要它：
    - 增量优化时，我们只想“减掉旧 pre-token 的贡献，再加上新 pre-token 的贡献”。
    - 这样不用每一轮都重新扫描所有 pre-token。
    """
    local_counts: Counter[TokenPair] = Counter()
    n = len(pretoken)
    for i in range(n - 1):
        local_counts[(pretoken[i], pretoken[i + 1])] += count
    return local_counts


def build_pair_indexes(
    pretoken_counts: dict[Pretoken, int],
) -> tuple[Counter[TokenPair], dict[TokenPair, set[Pretoken]]]:
    """一次性建立两个索引。

    pair_counts:
    - 类型：Counter[TokenPair]
    - 含义：当前每个 pair 在整个训练语料中出现多少次。

    pair_to_pretokens:
    - 类型：dict[TokenPair, set[Pretoken]]
    - 含义：某个 pair 出现在哪些 pre-token 里。
    - set 是集合，特点是去重，适合表示“有哪些 pre-token”。

    Python 语法提示：
    - set() 创建空集合。
    - dict.setdefault(key, default) 表示：如果 key 不存在，先放入 default；然后返回对应 value。
    - pair_to_pretokens.setdefault(pair, set()).add(pretoken)
      表示把 pretoken 加入这个 pair 对应的集合。
    """
    pair_counts: Counter[TokenPair] = Counter()
    pair_to_pretokens: dict[TokenPair, set[Pretoken]] = {}

    for pretoken, count in pretoken_counts.items():
        local_counts = count_pairs_in_one_pretoken(pretoken, count)
        pair_counts.update(local_counts)

        for pair in local_counts:
            pair_to_pretokens.setdefault(pair, set()).add(pretoken)

    return pair_counts, pair_to_pretokens


def build_pair_heap(pair_counts: Counter[TokenPair]) -> list[_PairHeapEntry]:
    """从当前频率表建立最大优先队列。"""
    heap = [_PairHeapEntry(count=count, pair=pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)
    return heap


def pop_best_pair(
    pair_heap: list[_PairHeapEntry],
    pair_counts: Counter[TokenPair],
) -> TokenPair | None:
    """弹出当前仍有效的最高优先级 pair，自动跳过过期记录。"""
    while pair_heap:
        entry = heapq.heappop(pair_heap)
        if pair_counts.get(entry.pair, 0) == entry.count and entry.count > 0:
            return entry.pair
    return None


def remove_nonpositive_pairs(
    pair_counts: Counter[TokenPair],
    pair_to_pretokens: dict[TokenPair, set[Pretoken]],
) -> None:
    """删除计数已经小于等于 0 的 pair。

    为什么需要：
    - 增量更新时，我们会对旧 pair 做减法。
    - Counter 里计数变成 0 或负数时，key 可能仍然存在。
    - choose_best_pair 前需要清理这些无效 pair。

    Python 语法提示：
    - list(pair_counts.items()) 是为了复制一份列表再遍历。
    - 如果边遍历 dict 边删除 key，Python 会报错。
    """
    for pair, count in list(pair_counts.items()):
        if count <= 0:
            del pair_counts[pair]
            pair_to_pretokens.pop(pair, None)


def add_pair_contribution(
    pair_counts: Counter[TokenPair],
    pair_to_pretokens: dict[TokenPair, set[Pretoken]],
    pretoken: Pretoken,
    count: int,
) -> set[TokenPair]:
    """把一个 pre-token 对 pair 索引的贡献加回去。"""
    local_counts = count_pairs_in_one_pretoken(pretoken, count)
    pair_counts.update(local_counts)
    for pair in local_counts:
        pair_to_pretokens.setdefault(pair, set()).add(pretoken)
    return set(local_counts)


def subtract_pair_contribution(
    pair_counts: Counter[TokenPair],
    pair_to_pretokens: dict[TokenPair, set[Pretoken]],
    pretoken: Pretoken,
    count: int,
) -> set[TokenPair]:
    """把一个旧 pre-token 对 pair 索引的贡献减掉。"""
    local_counts = count_pairs_in_one_pretoken(pretoken, count)
    for pair, pair_count in local_counts.items():
        pair_counts[pair] -= pair_count
        holders = pair_to_pretokens.get(pair)
        if holders is not None:
            holders.discard(pretoken)
            if not holders:
                pair_to_pretokens.pop(pair, None)
    return set(local_counts)


def choose_best_pair(pair_counts: Counter[TokenPair]) -> TokenPair:
    """选择下一次要 merge 的 pair。

    测试要求：
    - 第一优先级：频率最高。
    - 如果频率并列：选择字典序更大的 pair。

    Python 语法/数据结构提示：
    - max(...) 可以从一堆候选里选最大值。
    - key=... 可以告诉 max 按什么规则比较。
    - pair_counts[pair] 可以取出某个 pair 的出现次数。

    建议：
    - 这个函数应该非常短，方便你手工测试。
    """
    best_pair = max(
        pair_counts,
        key=lambda pair: (pair_counts[pair], pair),
    )
    return best_pair


def pretoken_contains_pair(pretoken: Pretoken, pair: TokenPair) -> bool:
    """判断 pretoken 里是否存在相邻的 pair。"""
    first, second = pair

    # 扫描每个相邻位置，range 到 len - 1 是为了让 i + 1 不越界
    for i in range(len(pretoken) - 1):
        if pretoken[i] == first and pretoken[i + 1] == second:
            return True  # 找到一处就够了，立刻返回

    return False  # 全扫完都没找到


def merge_pair_in_all_pretokens(
    pretoken_counts: dict[Pretoken, int],
    pair_to_merge: TokenPair,
) -> dict[Pretoken, int]:
    first, second = pair_to_merge
    merged_token = first + second

    new_counts: dict[Pretoken, int] = {}

    for pretoken, count in pretoken_counts.items():
        # 优化点：不含目标 pair 的 pre-token 原样搬过去，跳过重建
        if not pretoken_contains_pair(pretoken, pair_to_merge):
            if pretoken in new_counts:
                new_counts[pretoken] = new_counts[pretoken] + count
            else:
                new_counts[pretoken] = count
            continue

        # 含有目标 pair 的才真正扫描并合并
        new_tokens = []
        i = 0

        while i < len(pretoken):
            if i < len(pretoken) - 1 and pretoken[i] == first and pretoken[i + 1] == second:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(pretoken[i])
                i += 1

        new_pretoken = tuple(new_tokens)

        if new_pretoken in new_counts:
            new_counts[new_pretoken] = new_counts[new_pretoken] + count
        else:
            new_counts[new_pretoken] = count

    return new_counts


def merge_one_pretoken(pretoken: Pretoken, pair_to_merge: TokenPair) -> Pretoken:
    """只合并一个 pre-token。

    这个函数和 merge_pair_in_all_pretokens 的内部扫描逻辑一样，
    但它只处理一个 pre-token，方便增量版本复用。
    """
    first, second = pair_to_merge
    merged_token = first + second
    n = len(pretoken)
    new_tokens = []
    i = 0

    while i < n:
        if i + 1 < n and pretoken[i] == first and pretoken[i + 1] == second:
            new_tokens.append(merged_token)
            i += 2
        else:
            new_tokens.append(pretoken[i])
            i += 1

    return tuple(new_tokens)


def merge_pair_incremental(
    pretoken_counts: dict[Pretoken, int],
    pair_counts: Counter[TokenPair],
    pair_to_pretokens: dict[TokenPair, set[Pretoken]],
    pair_to_merge: TokenPair,
) -> tuple[dict[Pretoken, int], Counter[TokenPair], dict[TokenPair, set[Pretoken]]]:
    """增量 merge：只更新受 pair_to_merge 影响的 pre-token。

    核心思想：
    - old pre-token 如果不包含 pair_to_merge，就完全不用动。
    - old pre-token 如果包含 pair_to_merge：
      1. 从 pretoken_counts 中拿到它的 count。
      2. 从 pair_counts / pair_to_pretokens 中减掉旧贡献。
      3. 合并得到 new_pretoken。
      4. 从 pretoken_counts 中删除旧 pre-token，加入新 pre-token。
      5. 给 pair_counts / pair_to_pretokens 加上新贡献。

    注意：
    - 多个旧 pre-token 可能合并成同一个 new_pretoken，所以 pretoken_counts 要用 +=。
    - affected_pretokens 必须复制成 list，否则后面修改 set/dict 时可能影响遍历。
    """
    merge_pair_incremental_with_changes(
        pretoken_counts=pretoken_counts,
        pair_counts=pair_counts,
        pair_to_pretokens=pair_to_pretokens,
        pair_to_merge=pair_to_merge,
    )
    return pretoken_counts, pair_counts, pair_to_pretokens


def merge_pair_incremental_with_changes(
    pretoken_counts: dict[Pretoken, int],
    pair_counts: Counter[TokenPair],
    pair_to_pretokens: dict[TokenPair, set[Pretoken]],
    pair_to_merge: TokenPair,
) -> set[TokenPair]:
    """批量更新受一次 merge 影响的 pre-token，并返回发生变化的 pair。"""
    transformations: dict[Pretoken, tuple[Pretoken, int]] = {}
    for old_pretoken in tuple(pair_to_pretokens.get(pair_to_merge, ())):
        count = pretoken_counts.get(old_pretoken, 0)
        if count <= 0:
            continue
        new_pretoken = merge_one_pretoken(old_pretoken, pair_to_merge)
        if new_pretoken != old_pretoken:
            transformations[old_pretoken] = (new_pretoken, count)

    if not transformations:
        pair_counts.pop(pair_to_merge, None)
        pair_to_pretokens.pop(pair_to_merge, None)
        return {pair_to_merge}

    old_pretokens = set(transformations)
    destination_pretokens = {new_pretoken for new_pretoken, _ in transformations.values()}

    # 若目标 pre-token 原本已存在，先移除旧贡献，最后按合并后的总 count 一次性加回。
    existing_destination_counts = {
        pretoken: pretoken_counts.get(pretoken, 0)
        for pretoken in destination_pretokens
        if pretoken not in old_pretokens
    }
    pretokens_to_remove = old_pretokens | {
        pretoken for pretoken, count in existing_destination_counts.items() if count > 0
    }

    changed_pairs: set[TokenPair] = set()
    for pretoken in pretokens_to_remove:
        count = pretoken_counts.get(pretoken, 0)
        if count > 0:
            changed_pairs.update(
                subtract_pair_contribution(
                    pair_counts=pair_counts,
                    pair_to_pretokens=pair_to_pretokens,
                    pretoken=pretoken,
                    count=count,
                )
            )

    for pretoken in pretokens_to_remove:
        pretoken_counts.pop(pretoken, None)

    merged_counts: Counter[Pretoken] = Counter(existing_destination_counts)
    for new_pretoken, count in transformations.values():
        merged_counts[new_pretoken] += count

    for pretoken, count in merged_counts.items():
        pretoken_counts[pretoken] = count
        changed_pairs.update(
            add_pair_contribution(
                pair_counts=pair_counts,
                pair_to_pretokens=pair_to_pretokens,
                pretoken=pretoken,
                count=count,
            )
        )

    for pair in changed_pairs:
        if pair_counts.get(pair, 0) <= 0:
            pair_counts.pop(pair, None)
            if not pair_to_pretokens.get(pair):
                pair_to_pretokens.pop(pair, None)
    return changed_pairs
