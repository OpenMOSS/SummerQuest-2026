import os
import regex as re
from collections import Counter, defaultdict
from heapq import heapify, heappop, heappush
from multiprocessing import Pool, cpu_count
from pathlib import Path

GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _find_chunk_boundaries(
    file, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """将文件按特殊 token 边界分成若干块，返回字节偏移边界列表。"""
    assert isinstance(split_special_token, bytes), "特殊 token 必须以 bytes 表示"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks if desired_num_chunks > 0 else file_size
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


class _MaxPairEntry:
    """大根堆条目：count 越大越靠前；count 相同时 bytes 字典序越大越靠前。"""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: "_MaxPairEntry") -> bool:
        return (self.count, self.pair) > (other.count, other.pair)


def _count_pretokens(text: str, special_tokens: tuple[str, ...]) -> Counter[bytes]:
    """统计一段文本中每个 pre-token bytes 出现的次数。
    special tokens 会作为独立 pre-token 保留。
    """
    frequencies: Counter[bytes] = Counter()
    if not special_tokens:
        for match in GPT2_PATTERN.finditer(text):
            frequencies[match.group().encode("utf-8")] += 1
        return frequencies

    splitter = re.compile(
        "|".join(re.escape(token) for token in special_tokens)
    )
    start = 0
    for special_match in splitter.finditer(text):
        for match in GPT2_PATTERN.finditer(text, start, special_match.start()):
            frequencies[match.group().encode("utf-8")] += 1
        frequencies[special_match.group().encode("utf-8")] += 1
        start = special_match.end()
    for match in GPT2_PATTERN.finditer(text, start):
        frequencies[match.group().encode("utf-8")] += 1
    return frequencies


def _count_pretokens_chunk(
    input_path: str, start: int, end: int, special_tokens: tuple[str, ...]
) -> Counter[bytes]:
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens(text, special_tokens)


def _pretoken_frequencies(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
) -> Counter[bytes]:
    """多进程统计整个文件的 pre-token 频率。"""
    specials = tuple(special_tokens)
    path = Path(input_path)
    file_size = path.stat().st_size
    if file_size == 0:
        return Counter()

    if num_processes <= 1 or not special_tokens:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return _count_pretokens(text, specials)

    # 块数多于 worker 数，平衡不均匀文档并降低单进程内存峰值
    split_token = max(special_tokens, key=len).encode("utf-8")
    boundaries = _find_chunk_boundaries(
        open(path, "rb"), num_processes * 4, split_token
    )
    ranges = [
        (start, end) for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]
    if len(ranges) <= 1:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return _count_pretokens(text, specials)

    frequencies: Counter[bytes] = Counter()
    n_workers = min(num_processes, len(ranges))
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(
            _count_pretokens_chunk,
            [(str(path), start, end, specials) for start, end in ranges],
        )
    for res in results:
        frequencies.update(res)
    return frequencies


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 byte-level BPE（聚合 + 精确位置索引 + 链表）。"""
    special_bytes = [token.encode("utf-8") for token in special_tokens]
    if vocab_size < 256 + len(special_bytes):
        raise ValueError("vocab_size 不足以容纳 byte tokens 和 special tokens")

    # ---- 1. 多进程预分词，得到 <pre-token byte ids tuple, frequency> ----
    num_processes = kwargs.get("num_processes", min(cpu_count(), 8))
    print(f"[train_bpe] counting unique pre-tokens with "
          f"{num_processes} workers...", flush=True)
    pretoken_frequencies = _pretoken_frequencies(
        input_path, special_tokens, num_processes
    )

    # ---- 2. 初始化词表 ----
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_bytes:
        if token not in vocab.values():
            vocab[len(vocab)] = token
    special_bytes_to_id = {v: k for k, v in vocab.items() if v in special_bytes}
    merges: list[tuple[bytes, bytes]] = []

    # 转换为 byte-id tuple -> frequency，special token 作为单元素 tuple 保留
    unique_counts: dict[tuple[int, ...], int] = {}
    total_instances = 0
    total_bytes = 0
    for pt_bytes, freq in pretoken_frequencies.items():
        if pt_bytes in special_bytes_to_id:
            seq = (special_bytes_to_id[pt_bytes],)
        else:
            seq = tuple(pt_bytes)
        unique_counts[seq] = unique_counts.get(seq, 0) + freq
        total_instances += freq
        total_bytes += len(pt_bytes) * freq
    print(f"[train_bpe] {total_bytes/1024/1024:.1f} MB text, "
          f"{total_instances} pre-token instances, "
          f"{len(unique_counts)} unique pre-tokens", flush=True)
    del pretoken_frequencies

    if not unique_counts:
        return vocab, merges

    # ---- 3. 为每个 unique pre-token 建立链表 ----
    words_token: list[list[int]] = []
    words_prev: list[list[int]] = []
    words_next: list[list[int]] = []
    words_deleted: list[list[bool]] = []
    words_head: list[int] = []
    word_weight: list[int] = []

    pair_counts: dict[tuple[int, int], int] = {}
    pair_positions: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add_pair(word_idx: int, left_idx: int, pair: tuple[int, int]) -> None:
        pair_counts[pair] = pair_counts.get(pair, 0) + word_weight[word_idx]
        pair_positions.setdefault(pair, []).append((word_idx, left_idx))

    for seq_tuple, weight in unique_counts.items():
        word_idx = len(words_token)
        seq = list(seq_tuple)
        n = len(seq)
        words_token.append(seq)
        words_prev.append(list(range(-1, n - 1)))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        words_next.append(nxt)
        words_deleted.append([False] * n)
        words_head.append(0)
        word_weight.append(weight)

        left_idx = 0
        while left_idx != -1:
            right_idx = words_next[word_idx][left_idx]
            if right_idx == -1:
                break
            pair = (words_token[word_idx][left_idx], words_token[word_idx][right_idx])
            add_pair(word_idx, left_idx, pair)
            left_idx = right_idx

    del unique_counts

    # ---- 4. 大根堆 ----
    class _HeapEntry:
        __slots__ = ("count", "pair", "byte_pair")

        def __init__(self, count: int, pair: tuple[int, int]) -> None:
            self.count = count
            self.pair = pair
            self.byte_pair = (vocab[pair[0]], vocab[pair[1]])

        def __lt__(self, other: "_HeapEntry") -> bool:
            if self.count != other.count:
                return self.count > other.count
            return self.byte_pair > other.byte_pair

    heap: list[_HeapEntry] = []

    def push_pair(pair: tuple[int, int]) -> None:
        count = pair_counts.get(pair, 0)
        if count > 0:
            heappush(heap, _HeapEntry(count, pair))

    for pair in pair_counts:
        push_pair(pair)

    # ---- 5. 主循环 ----
    import time as _time
    _merge_t0 = _time.time()
    target_merges = vocab_size - len(vocab)

    while len(vocab) < vocab_size and pair_counts:
        # 弹出 stale 条目
        while heap:
            entry = heap[0]
            current_count = pair_counts.get(entry.pair, 0)
            if current_count == entry.count and current_count > 0:
                break
            heappop(heap)
        else:
            break

        entry = heappop(heap)
        best_pair = entry.pair
        best_count = entry.count
        if best_count <= 0:
            break

        _n_merges = len(merges)
        if _n_merges < 5 or _n_merges % 200 == 0:
            _elapsed = _time.time() - _merge_t0
            _pct = 100.0 * _n_merges / target_merges if target_merges > 0 else 0
            print(f"[train_bpe] merge {_n_merges}/{target_merges} ({_pct:.1f}%) | "
                  f"vocab={len(vocab)} | pairs={len(pair_counts)} | "
                  f"best_pair_count={best_count} | "
                  f"{_elapsed:.1f}s elapsed", flush=True)

        a, b = best_pair
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        positions = pair_positions.get(best_pair, [])
        pair_positions[best_pair] = []
        pair_counts[best_pair] = 0

        changed_pairs: set[tuple[int, int]] = set()

        for word_idx, left_idx in positions:
            if words_deleted[word_idx][left_idx]:
                continue
            right_idx = words_next[word_idx][left_idx]
            if right_idx == -1:
                continue
            if words_deleted[word_idx][right_idx]:
                continue
            if words_token[word_idx][right_idx] != b:
                continue

            prev_idx = words_prev[word_idx][left_idx]
            next_idx = words_next[word_idx][right_idx]
            prev_token = words_token[word_idx][prev_idx] if prev_idx != -1 else None
            next_token = words_token[word_idx][next_idx] if next_idx != -1 else None
            weight = word_weight[word_idx]

            # 移除旧相邻 pair
            if prev_idx != -1:
                old_pair = (prev_token, a)
                pair_counts[old_pair] = pair_counts.get(old_pair, 0) - weight
                changed_pairs.add(old_pair)
            if next_idx != -1:
                old_pair = (b, next_token)
                pair_counts[old_pair] = pair_counts.get(old_pair, 0) - weight
                changed_pairs.add(old_pair)

            # 标记旧节点删除并创建新合并节点
            words_deleted[word_idx][left_idx] = True
            words_deleted[word_idx][right_idx] = True

            new_idx = len(words_token[word_idx])
            words_token[word_idx].append(new_token_id)
            words_prev[word_idx].append(prev_idx)
            words_next[word_idx].append(next_idx)
            words_deleted[word_idx].append(False)

            # 重新连接邻居
            if prev_idx != -1:
                words_next[word_idx][prev_idx] = new_idx
            else:
                words_head[word_idx] = new_idx
            if next_idx != -1:
                words_prev[word_idx][next_idx] = new_idx

            # 添加新相邻 pair
            if prev_idx != -1:
                new_pair = (prev_token, new_token_id)
                add_pair(word_idx, prev_idx, new_pair)
                changed_pairs.add(new_pair)
            if next_idx != -1:
                new_pair = (new_token_id, next_token)
                add_pair(word_idx, new_idx, new_pair)
                changed_pairs.add(new_pair)

        # 推送变化 pair 并清理 count 为 0 的条目
        for pair in changed_pairs:
            count = pair_counts.get(pair, 0)
            if count <= 0:
                pair_counts.pop(pair, None)
                if pair in pair_positions:
                    pair_positions[pair] = []
            else:
                push_pair(pair)

    return (vocab, merges)


class BPETokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id_to_bytes: dict[int, bytes] = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in vocab.items()}

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []
        self.special_tokens_sorted: list[str] = sorted(
            self.special_tokens, key=len, reverse=True
        )

        if self.special_tokens_sorted:
            pattern = "(" + "|".join(
                re.escape(tok) for tok in self.special_tokens_sorted
            ) + ")"
            self._special_split_re: re.Pattern | None = re.compile(pattern)
        else:
            self._special_split_re = None

        self._special_to_id: dict[str, int] = {}
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes in self.bytes_to_id:
                self._special_to_id[tok] = self.bytes_to_id[tok_bytes]

    def _split_special(self, text: str) -> list[tuple[str, bool]]:
        if self._special_split_re is None:
            return [(text, False)]
        parts = self._special_split_re.split(text)
        result: list[tuple[str, bool]] = []
        for part in parts:
            if not part:
                continue
            is_special = part in self._special_to_id
            result.append((part, is_special))
        return result

    def _merge_pre_token(self, pre_token_bytes: bytes) -> list[int]:
        tokens: list[bytes] = [bytes([b]) for b in pre_token_bytes]
        while len(tokens) >= 2:
            best_rank: int | None = None
            best_idx: int = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens[best_idx : best_idx + 2] = [merged]
        return [self.bytes_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment, is_special in self._split_special(text):
            if is_special:
                ids.append(self._special_to_id[segment])
                continue
            for m in GPT2_PATTERN.finditer(segment):
                pre_token_bytes = m.group().encode("utf-8")
                ids.extend(self._merge_pre_token(pre_token_bytes))
        return ids

    def decode(self, ids: list[int]) -> str:
        buf: list[bytes] = []
        for _id in ids:
            buf.append(self.id_to_bytes[_id])
        full_bytes = b"".join(buf)
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        yield from self._encode_iterable_impl(iterable)

    def _encode_iterable_impl(self, iterable):
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            special_spans: list[tuple[int, int, str]] = []
            if self._special_split_re is not None:
                for m in self._special_split_re.finditer(buffer):
                    t = m.group()
                    if t in self._special_to_id:
                        special_spans.append((m.start(), m.end(), t))

            cut_pos = 0
            if special_spans:
                last_special_start, last_special_end, last_special_text = special_spans[-1]
                safe_block = buffer[:last_special_end]
                yield from self._encode_small_block(safe_block)
                cut_pos = last_special_end

            tail = buffer[cut_pos:]
            matches = list(GPT2_PATTERN.finditer(tail))
            if len(matches) >= 2:
                process_up_to_in_tail = matches[-1].start()
                processable_tail = tail[:process_up_to_in_tail]
                yield from self._encode_small_block(processable_tail)
                cut_pos += process_up_to_in_tail

            buffer = buffer[cut_pos:]

        if buffer:
            yield from self._encode_small_block(buffer)

    def _encode_small_block(self, text: str):
        for segment, is_special in self._split_special(text):
            if is_special:
                yield self._special_to_id[segment]
                continue
            for m in GPT2_PATTERN.finditer(segment):
                pt_bytes = m.group().encode("utf-8")
                for _id in self._merge_pre_token(pt_bytes):
                    yield _id
