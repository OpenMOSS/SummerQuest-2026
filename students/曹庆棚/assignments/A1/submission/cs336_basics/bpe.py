from __future__ import annotations

import heapq
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import BinaryIO

import regex
from tqdm.auto import tqdm


Token = bytes
Pair = tuple[Token, Token]
Word = tuple[Token, ...]
EncodedPretokenCounts = Counter[bytes]


# GPT-2 风格的 pre-tokenization 正则。
# regex 包支持 \p{L}、\p{N} 等 Unicode property。
GPT2_PRETOKEN_PATTERN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class _ReverseLexicographicPair:
    """
    heapq 默认是最小堆。

    对频次相同的 pair，题目要求选择字典序更大的 pair，
    因此这里反转 pair 的比较方向。
    """

    __slots__ = ("pair",)

    def __init__(self, pair: Pair):
        self.pair = pair

    def __lt__(self, other: _ReverseLexicographicPair) -> bool:
        return self.pair > other.pair


def _deduplicate_special_tokens(special_tokens: list[str]) -> list[str]:
    """去重并保留 special token 的原始顺序。"""
    return list(dict.fromkeys(special_tokens))


def _iter_regions_without_special_tokens(
    text: str,
    special_tokens: list[str],
) -> Iterable[str]:
    """
    将 special tokens 视为不可跨越的硬边界。

    返回 special token 之间的普通文本区域，special token 本身不参与
    普通 pre-token pair 统计。
    """
    nonempty_special_tokens = [token for token in special_tokens if token]

    if not nonempty_special_tokens:
        yield text
        return

    # 较长的 special token 放在前面，正确处理互相重叠的 token。
    ordered_tokens = sorted(
        nonempty_special_tokens,
        key=len,
        reverse=True,
    )

    special_pattern = regex.compile("|".join(regex.escape(token) for token in ordered_tokens))

    previous_end = 0

    for match in special_pattern.finditer(text):
        if match.start() > previous_end:
            yield text[previous_end : match.start()]

        previous_end = match.end()

    if previous_end < len(text):
        yield text[previous_end:]


def _count_pretokens(
    text: str,
    special_tokens: list[str],
    byte_tokens: tuple[bytes, ...],
) -> Counter[Word]:
    """
    执行 GPT-2 pre-tokenization，并统计每种 pre-token 的出现次数。

    一个 Word 是 byte token 构成的 tuple，例如：

        "the".encode("utf-8")
        -> (b"t", b"h", b"e")
    """
    pretoken_counts: Counter[Word] = Counter()

    for region in _iter_regions_without_special_tokens(
        text=text,
        special_tokens=special_tokens,
    ):
        for match in GPT2_PRETOKEN_PATTERN.finditer(region):
            encoded = match.group(0).encode("utf-8")

            word = tuple(byte_tokens[byte_value] for byte_value in encoded)

            if word:
                pretoken_counts[word] += 1

    return pretoken_counts


def _count_encoded_pretokens(
    text: str,
    special_tokens: list[str],
) -> EncodedPretokenCounts:
    """Count pre-tokens as compact UTF-8 byte strings.

    Keeping one ``bytes`` object per distinct pre-token is substantially more
    memory-efficient than keeping a tuple of one-byte objects while the corpus
    is being scanned. The byte tuples needed by the merge algorithm are built
    only once, after all chunk counters have been combined.
    """
    pretoken_counts: EncodedPretokenCounts = Counter()

    for region in _iter_regions_without_special_tokens(text=text, special_tokens=special_tokens):
        for match in GPT2_PRETOKEN_PATTERN.finditer(region):
            encoded = match.group(0).encode("utf-8")
            if encoded:
                pretoken_counts[encoded] += 1

    return pretoken_counts


def _find_next_boundary(
    source: BinaryIO,
    start: int,
    split_special_token: bytes,
    file_size: int,
    scan_size: int = 1024 * 1024,
) -> int:
    """Find the next complete split token at or after a byte offset."""
    source.seek(start)
    overlap = b""
    position = start
    overlap_size = max(0, len(split_special_token) - 1)

    while position < file_size:
        block = source.read(min(scan_size, file_size - position))
        if not block:
            break
        searchable = overlap + block
        found_at = searchable.find(split_special_token)
        if found_at >= 0:
            return position - len(overlap) + found_at
        overlap = searchable[-overlap_size:] if overlap_size else b""
        position += len(block)

    return file_size


def _find_chunk_boundaries(
    input_path: Path,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Choose independent byte ranges whose boundaries are special tokens.

    A boundary is placed at the beginning of a special token. Consequently,
    ordinary GPT-2 pre-tokens cannot cross it, and summing per-range counters is
    exactly equivalent to counting the entire corpus at once.
    """
    file_size = input_path.stat().st_size
    if file_size == 0:
        return [0]

    desired_num_chunks = max(1, desired_num_chunks)
    if desired_num_chunks == 1:
        return [0, file_size]

    boundaries = {0, file_size}
    with input_path.open("rb") as source:
        for chunk_index in range(1, desired_num_chunks):
            target = file_size * chunk_index // desired_num_chunks
            boundary = _find_next_boundary(source, target, split_special_token, file_size)
            boundaries.add(boundary)

    return sorted(boundaries)


def _count_file_range(task: tuple[str, int, int, tuple[str, ...]]) -> tuple[EncodedPretokenCounts, int]:
    """Read and pre-tokenize one independently splittable corpus range."""
    input_path, start, end, special_tokens = task
    with Path(input_path).open("rb") as source:
        source.seek(start)
        raw_text = source.read(end - start)
    text = raw_text.decode("utf-8")
    return _count_encoded_pretokens(text=text, special_tokens=list(special_tokens)), len(raw_text)


def _count_corpus_pretokens(
    input_path: Path,
    special_tokens: list[str],
    *,
    num_processes: int,
    chunk_size_bytes: int,
    progress: bool,
) -> EncodedPretokenCounts:
    """Count a corpus incrementally, optionally using worker processes."""
    file_size = input_path.stat().st_size
    nonempty_special_tokens = [token.encode("utf-8") for token in special_tokens if token]

    # Exact arbitrary byte splitting is not possible because GPT-2 pre-tokens
    # may span a chunk boundary. Without a special-token delimiter, preserve
    # exact behavior by using one range.
    if nonempty_special_tokens:
        desired_num_chunks = max(num_processes, math.ceil(file_size / chunk_size_bytes)) if file_size else 1
        # Splitting on a longest special token also remains correct when one
        # configured token is a suffix of another; a shorter suffix could put a
        # boundary in the middle of the longer special token.
        split_special_token = max(nonempty_special_tokens, key=len)
        boundaries = _find_chunk_boundaries(input_path, desired_num_chunks, split_special_token)
    else:
        boundaries = [0, file_size] if file_size else [0]

    tasks = [
        (str(input_path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]
    combined: EncodedPretokenCounts = Counter()
    worker_count = min(num_processes, len(tasks)) if tasks else 1

    with tqdm(
        total=file_size,
        desc=f"BPE pre-token count ({worker_count} worker{'s' if worker_count != 1 else ''})",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not progress,
    ) as count_progress:
        if worker_count == 1:
            for task in tasks:
                counts, bytes_read = _count_file_range(task)
                combined.update(counts)
                count_progress.update(bytes_read)
            return combined

        # Keep only a small bounded number of futures alive. A completed future
        # owns its returned Counter, so submitting every range at once can retain
        # several large counters in the parent process.
        task_iterator = iter(tasks)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pending: set[Future[tuple[EncodedPretokenCounts, int]]] = set()
            for _ in range(min(worker_count, len(tasks))):
                pending.add(executor.submit(_count_file_range, next(task_iterator)))

            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    counts, bytes_read = future.result()
                    combined.update(counts)
                    count_progress.update(bytes_read)
                    try:
                        pending.add(executor.submit(_count_file_range, next(task_iterator)))
                    except StopIteration:
                        pass

    return combined


def _pretoken_cache_metadata(input_path: Path, special_tokens: list[str]) -> dict[str, object]:
    stat = input_path.stat()
    return {
        "format_version": 1,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "special_tokens": special_tokens,
        "pretoken_pattern": GPT2_PRETOKEN_PATTERN.pattern,
    }


def _load_pretoken_cache(
    cache_path: Path,
    input_path: Path,
    special_tokens: list[str],
) -> EncodedPretokenCounts | None:
    """Load a trusted, self-generated pre-token Counter when metadata matches."""
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as source:
        payload = pickle.load(source)  # noqa: S301 - this cache is explicitly local/trusted
    if not isinstance(payload, dict) or payload.get("metadata") != _pretoken_cache_metadata(input_path, special_tokens):
        return None
    counts = payload.get("counts")
    return counts if isinstance(counts, Counter) else None


def _write_pretoken_cache(
    cache_path: Path,
    input_path: Path,
    special_tokens: list[str],
    counts: EncodedPretokenCounts,
) -> None:
    """Atomically persist a trusted pre-token Counter for repeated BPE runs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as destination:
            pickle.dump(
                {
                    "metadata": _pretoken_cache_metadata(input_path, special_tokens),
                    "counts": counts,
                },
                destination,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _count_adjacent_pairs(word: Word) -> Counter[Pair]:
    """统计一个 pre-token 内部的相邻 pair，保留重复出现次数。"""
    return Counter(zip(word, word[1:]))


def _merge_pair_in_word(
    word: Word,
    pair: Pair,
    merged_token: Token,
) -> Word:
    """
    从左到右合并一个 word 中所有不重叠的指定 pair。

    例如：

        word = (b"a", b"a", b"a")
        pair = (b"a", b"a")

    结果是：

        (b"aa", b"a")
    """
    result: list[Token] = []
    index = 0

    while index < len(word):
        if index + 1 < len(word) and word[index] == pair[0] and word[index + 1] == pair[1]:
            result.append(merged_token)
            index += 2
        else:
            result.append(word[index])
            index += 1

    return tuple(result)


def _merge_pair_in_word_with_deltas(
    word: Word,
    pair: Pair,
    merged_token: Token,
) -> tuple[Word, Counter[Pair]]:
    """Merge a pair and return only the adjacency-count changes.

    An occurrence can change at most the pair itself plus its immediate left
    and right neighbors. Interior pairs elsewhere in the pre-token are copied
    unchanged, so recounting the complete old and new words is unnecessary.
    """
    result: list[Token] = []
    merged_output_indices: list[int] = []
    changed_old_pair_indices: set[int] = set()
    index = 0

    while index < len(word):
        if index + 1 < len(word) and word[index] == pair[0] and word[index + 1] == pair[1]:
            merged_output_indices.append(len(result))
            result.append(merged_token)
            for pair_index in (index - 1, index, index + 1):
                if 0 <= pair_index < len(word) - 1:
                    changed_old_pair_indices.add(pair_index)
            index += 2
        else:
            result.append(word[index])
            index += 1

    if not merged_output_indices:
        return word, Counter()

    new_word = tuple(result)
    deltas: Counter[Pair] = Counter()
    for pair_index in changed_old_pair_indices:
        deltas[(word[pair_index], word[pair_index + 1])] -= 1

    changed_new_pair_indices: set[int] = set()
    for output_index in merged_output_indices:
        for pair_index in (output_index - 1, output_index):
            if 0 <= pair_index < len(new_word) - 1:
                changed_new_pair_indices.add(pair_index)
    for pair_index in changed_new_pair_indices:
        deltas[(new_word[pair_index], new_word[pair_index + 1])] += 1

    # Counter keeps zero-valued keys after cancellation, but callers should
    # touch only pairs whose multiplicity really changed.
    return new_word, Counter({changed_pair: delta for changed_pair, delta in deltas.items() if delta})


def _push_pair_to_heap(
    heap: list[tuple[int, _ReverseLexicographicPair, Pair]],
    pair: Pair,
    count: int,
) -> None:
    if count <= 0:
        return

    heapq.heappush(
        heap,
        (
            -count,
            _ReverseLexicographicPair(pair),
            pair,
        ),
    )


def _pop_valid_pair(
    heap: list[tuple[int, _ReverseLexicographicPair, Pair]],
    pair_counts: Counter[Pair],
) -> Pair | None:
    """
    从 heap 中取出当前最高频 pair。

    pair count 更新后，heap 中可能保留旧记录，因此取出时需要检查
    记录是否仍与 pair_counts 中的当前值一致。
    """
    while heap:
        negative_count, _, pair = heapq.heappop(heap)
        recorded_count = -negative_count
        current_count = pair_counts.get(pair, 0)

        if current_count > 0 and current_count == recorded_count:
            return pair

    return None


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    progress: bool = False,
    num_processes: int = 1,
    chunk_size_bytes: int = 256 * 1024 * 1024,
    pretoken_cache_path: str | Path | None = None,
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 byte-level BPE tokenizer。

    Args:
        input_path:
            UTF-8 文本语料路径。
        vocab_size:
            最终词表大小，包含 256 个单 byte token 和 special tokens。
        special_tokens:
            加入词表、但不参加普通 pair merge 的特殊 token。
        progress:
            是否在 stderr 显示语料处理、pair 初始化和 merge 进度。
        num_processes:
            语料 pre-token 计数使用的进程数。BPE merge 阶段保持单进程，
            以维持确定性的 merge 顺序。
        chunk_size_bytes:
            pre-token 计数块的近似最大字节数。实际边界向后对齐到第一个
            special token，因此不会切断普通 pre-token。
        pretoken_cache_path:
            可选的本地可信 pickle cache。metadata 匹配时直接复用完整
            pre-token Counter；首次计数后通过临时文件原子写入。

    Returns:
        vocab:
            token ID 到 token bytes 的映射。
        merges:
            按创建顺序排列的 BPE merge rules。
    """
    special_tokens = _deduplicate_special_tokens(special_tokens)

    byte_tokens = tuple(bytes([value]) for value in range(256))

    vocab: dict[int, bytes] = {token_id: byte_tokens[token_id] for token_id in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size must be at least {len(vocab)} for 256 byte tokens and the supplied special tokens"
        )

    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    if chunk_size_bytes < 1:
        raise ValueError("chunk_size_bytes must be at least 1")

    input_path = Path(input_path)
    cache_path = Path(pretoken_cache_path) if pretoken_cache_path is not None else None
    encoded_pretoken_counts = (
        _load_pretoken_cache(cache_path, input_path, special_tokens) if cache_path is not None else None
    )
    if encoded_pretoken_counts is not None:
        if progress:
            tqdm.write(f"[BPE] loaded pre-token cache: {cache_path}", file=sys.stderr)
    else:
        if progress:
            tqdm.write(f"[BPE] streaming corpus: {input_path}", file=sys.stderr)
        encoded_pretoken_counts = _count_corpus_pretokens(
            input_path=input_path,
            special_tokens=special_tokens,
            num_processes=num_processes,
            chunk_size_bytes=chunk_size_bytes,
            progress=progress,
        )
        if cache_path is not None:
            if progress:
                tqdm.write(f"[BPE] writing pre-token cache: {cache_path}", file=sys.stderr)
            _write_pretoken_cache(cache_path, input_path, special_tokens, encoded_pretoken_counts)
    if progress:
        tqdm.write(f"[BPE] unique pre-tokens: {len(encoded_pretoken_counts):,}", file=sys.stderr)

    # 每种 pre-token 只保存一次；它在语料中的出现次数单独保存在 frequencies。
    words: list[Word] = [
        tuple(byte_tokens[byte_value] for byte_value in encoded) for encoded in encoded_pretoken_counts
    ]
    frequencies: list[int] = list(encoded_pretoken_counts.values())

    # 全局 pair 频率。
    pair_counts: Counter[Pair] = Counter()

    # pair -> 包含这个 pair 的 word ID 集合。
    # 每轮 merge 只更新真正包含目标 pair 的 words。
    pair_to_word_ids: dict[Pair, set[int]] = defaultdict(set)

    word_items = zip(words, frequencies, strict=True)
    for word_id, (word, frequency) in enumerate(
        tqdm(
            word_items,
            total=len(words),
            desc="BPE pair index",
            unit="word",
            dynamic_ncols=True,
            disable=not progress,
        )
    ):
        local_pair_counts = _count_adjacent_pairs(word)

        for pair, local_count in local_pair_counts.items():
            pair_counts[pair] += local_count * frequency
            pair_to_word_ids[pair].add(word_id)

    heap: list[tuple[int, _ReverseLexicographicPair, Pair]] = []

    for pair, count in pair_counts.items():
        _push_pair_to_heap(
            heap=heap,
            pair=pair,
            count=count,
        )

    merges: list[Pair] = []

    merge_target = vocab_size - len(vocab)
    with tqdm(
        total=merge_target,
        desc="BPE merges",
        unit="merge",
        dynamic_ncols=True,
        disable=not progress,
    ) as merge_progress:
        while len(vocab) < vocab_size:
            best_pair = _pop_valid_pair(
                heap=heap,
                pair_counts=pair_counts,
            )

            if best_pair is None:
                # 语料中已经没有可以继续合并的相邻 pair。
                break

            merged_token = best_pair[0] + best_pair[1]

            vocab[len(vocab)] = merged_token
            merges.append(best_pair)

            # 在修改索引前复制受影响的 word IDs。
            affected_word_ids = tuple(pair_to_word_ids.get(best_pair, set()))

            touched_pairs: set[Pair] = set()

            for word_id in affected_word_ids:
                old_word = words[word_id]
                frequency = frequencies[word_id]

                new_word, local_pair_deltas = _merge_pair_in_word_with_deltas(
                    word=old_word,
                    pair=best_pair,
                    merged_token=merged_token,
                )
                # The inverted index is a safe superset: disappeared pairs may
                # leave stale word IDs, but newly created pairs are always added.
                # Remove stale IDs opportunistically when their pair is chosen.
                if not local_pair_deltas:
                    pair_to_word_ids[best_pair].discard(word_id)
                    continue

                words[word_id] = new_word

                for changed_pair, local_delta in local_pair_deltas.items():
                    new_global_count = pair_counts.get(changed_pair, 0) + local_delta * frequency
                    if new_global_count > 0:
                        pair_counts[changed_pair] = new_global_count
                    else:
                        pair_counts.pop(changed_pair, None)
                        # A zero global count proves that every occurrence is
                        # gone, so its entire possibly-stale index can be freed.
                        pair_to_word_ids.pop(changed_pair, None)

                    if local_delta > 0:
                        pair_to_word_ids[changed_pair].add(word_id)

                    touched_pairs.add(changed_pair)

            # 将发生变化的 pair count 重新放入 heap。
            # 旧 heap entry 不立即删除，取出时通过 _pop_valid_pair 跳过。
            for pair in touched_pairs:
                _push_pair_to_heap(
                    heap=heap,
                    pair=pair,
                    count=pair_counts.get(pair, 0),
                )

            # Counts use lazy heap invalidation. Rebuild occasionally so old
            # entries cannot accumulate throughout a 32K-vocabulary run.
            if len(heap) > max(100_000, 4 * len(pair_counts)):
                heap = [
                    (-count, _ReverseLexicographicPair(pair), pair) for pair, count in pair_counts.items() if count > 0
                ]
                heapq.heapify(heap)

            merge_progress.update(1)

    return vocab, merges
