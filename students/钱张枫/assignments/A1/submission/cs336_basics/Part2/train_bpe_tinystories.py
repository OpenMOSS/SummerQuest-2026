from __future__ import annotations

import argparse
import heapq
import json
import os
import queue
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import psutil
import regex as re


"""
TinyStories 专用 BPE 训练脚本。

本文件刻意独立于 train_bpe.py，不从 train_bpe.py 导入任何函数。为了便于对比，
下面先复制 train_bpe.py 中的基础类型、校验、special token 切分、GPT-2 预分词、
词表初始化等通用逻辑；随后用带有 “TinyStories 修改点” 注释的代码替换原始
train_bpe.py 中不适合 2.1GB 语料的部分。

相对 train_bpe.py 的主要修改：
1. 原始 read_training_text 会整文件读入；这里改为按 <|endoftext|> 文档边界流式读取。
2. 原始 count_pretokens 单进程处理全部文本；这里改为多进程并行预分词，再聚合 Counter。
3. 原始 BPE 每轮 merge 都全量扫描所有 pretoken；这里使用 pair -> word 倒排索引和堆，只更新受影响的 pretoken。
4. 新增资源采样、最长 token 统计，以及 vocab/merges/summary JSON 序列化。
"""


GPT2_PRETOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
DEFAULT_OUTPUT_DIR = Path("artifacts/tinystories_bpe")
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_BATCH_BYTES = 64 * 1024 * 1024
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.2

Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
IdPair = tuple[int, int]

_worker_special_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingMetrics:
    total_seconds: float
    pretokenization_seconds: float
    bpe_training_seconds: float
    serialization_seconds: float
    peak_rss_mb: float
    unique_pretokens: int
    total_pretokens: int


@dataclass(frozen=True)
class DescendingPairKey:
    """TinyStories 修改点：堆内使用反向字典序，以匹配 BPE tie-break 规则。"""

    pair: Pair

    def __lt__(self, other: DescendingPairKey) -> bool:
        return self.pair > other.pair


class ProcessTreeMemorySampler:
    """TinyStories 修改点：采样当前进程和 worker 子进程的 RSS 峰值。"""

    def __init__(self, interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._errors: queue.SimpleQueue[Exception] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def __enter__(self) -> ProcessTreeMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop_event.set()
        self._thread.join()

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_rss_bytes = max(self.peak_rss_bytes, self._current_tree_rss_bytes())
            except Exception as exc:  # pragma: no cover - defensive sampler path
                self._errors.put(exc)
            self._stop_event.wait(self.interval_seconds)

    def _current_tree_rss_bytes(self) -> int:
        rss_bytes = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            try:
                rss_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss_bytes


# ===== 从 train_bpe.py 复制的基础逻辑：输入校验、special token 切分、预分词、词表初始化。 =====


def validate_train_bpe_inputs(
    input_path: str,
    vocab_size: int,
    special_tokens: Sequence[str],
) -> list[str]:
    """校验公开 API 的输入，并返回去重后的 special tokens。"""
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


def split_on_special_tokens(text: str, special_tokens: Sequence[str]) -> list[str]:
    """按 special token 将文本切分为普通片段，并丢弃 special token 片段。"""
    if not special_tokens:
        return [text]

    escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    special_token_pattern = "|".join(escaped_tokens)
    return [segment for segment in re.split(special_token_pattern, text) if segment]


def count_pretokens(text_segments: Iterable[str]) -> Counter[bytes]:
    """对文本片段做 GPT-2 风格预分词，并按 UTF-8 bytes 统计每个 pretoken。"""
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


# ===== TinyStories 修改点 1：替换原始整文件读取，按 <|endoftext|> 文档边界流式分批。 =====


def iter_special_token_batches(
    input_path: Path,
    special_token: str,
    target_batch_bytes: int,
) -> Iterator[bytes]:
    """产出以 special token 为硬边界的 byte batch，避免 merge 或预分词跨文档。"""
    delimiter = special_token.encode("utf-8")
    read_size = min(target_batch_bytes, 16 * 1024 * 1024)
    remainder = b""
    batch = bytearray()

    with input_path.open("rb") as input_file:
        while chunk := input_file.read(read_size):
            data = remainder + chunk
            documents = data.split(delimiter)
            remainder = documents.pop()

            for document in documents:
                batch.extend(document)
                batch.extend(delimiter)
                if len(batch) >= target_batch_bytes:
                    yield bytes(batch)
                    batch.clear()

    if remainder:
        batch.extend(remainder)
    if batch:
        yield bytes(batch)


# ===== TinyStories 修改点 2：替换原始单进程预分词，使用 worker 并行处理 batch。 =====


def initialize_worker(special_tokens: Sequence[str]) -> None:
    global _worker_special_tokens
    _worker_special_tokens = tuple(special_tokens)


def count_batch_pretokens(batch_bytes: bytes) -> Counter[bytes]:
    text = batch_bytes.decode("utf-8")
    text_segments = split_on_special_tokens(text, _worker_special_tokens)
    return count_pretokens(text_segments)


def count_pretokens_parallel(
    input_path: Path,
    special_tokens: Sequence[str],
    workers: int,
    batch_bytes: int,
) -> Counter[bytes]:
    pretoken_counts: Counter[bytes] = Counter()
    special_token = special_tokens[0]
    batches = iter_special_token_batches(input_path, special_token, batch_bytes)

    if workers == 1:
        initialize_worker(special_tokens)
        for batch_counts in map(count_batch_pretokens, batches):
            pretoken_counts.update(batch_counts)
        return pretoken_counts

    with Pool(processes=workers, initializer=initialize_worker, initargs=(tuple(special_tokens),)) as pool:
        for batch_counts in pool.imap_unordered(count_batch_pretokens, batches, chunksize=1):
            pretoken_counts.update(batch_counts)

    return pretoken_counts


# ===== TinyStories 修改点 3：替换原始每轮全量扫描，使用增量 pair 倒排索引训练 BPE。 =====


def train_bpe_from_pretoken_counts(
    word_counts: Mapping[bytes, int],
    vocab_size: int,
    special_tokens: Sequence[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """从已并行统计的 pretoken Counter 训练 BPE。"""
    token_counts = initialize_byte_tokens(word_counts)
    vocab = initialize_vocab(special_tokens)
    merges: list[Pair] = []
    return train_bpe_from_token_counts(token_counts, vocab, merges, vocab_size)


def train_bpe_from_token_counts(
    token_counts: Counter[Token],
    vocab: dict[int, bytes],
    merges: list[Pair],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[Pair]]:
    """维护 pair -> pretoken 的倒排索引，只重算受当前 best pair 影响的 pretoken。"""
    word_tokens: list[list[int]] = []
    word_weights: list[int] = []
    pair_counts: dict[IdPair, int] = {}
    pair_to_word_counts: dict[IdPair, dict[int, int]] = {}
    heap: list[tuple[int, DescendingPairKey, IdPair]] = []
    token_to_id = {token: token_id for token_id, token in vocab.items()}

    for token, count in token_counts.items():
        if count <= 0:
            continue
        word_id = len(word_tokens)
        token_ids = [token_to_id[part] for part in token]
        word_tokens.append(token_ids)
        word_weights.append(count)

        for pair, occurrences in count_token_id_pairs(token_ids).items():
            weighted_count = occurrences * count
            pair_counts[pair] = pair_counts.get(pair, 0) + weighted_count
            pair_to_word_counts.setdefault(pair, {})[word_id] = occurrences

    for pair, count in pair_counts.items():
        push_pair(heap, pair, count, vocab)

    while len(vocab) < vocab_size:
        best_pair = pop_best_pair(heap, pair_counts, vocab)
        if best_pair is None:
            break

        left_id, right_id = best_pair
        left_bytes = vocab[left_id]
        right_bytes = vocab[right_id]
        merged_token_id = len(vocab)
        vocab[merged_token_id] = left_bytes + right_bytes
        merges.append((left_bytes, right_bytes))

        affected_word_ids = list(pair_to_word_counts.get(best_pair, {}).keys())
        for word_id in affected_word_ids:
            old_token_ids = word_tokens[word_id]
            old_pair_counts = count_token_id_pairs(old_token_ids)
            if best_pair not in old_pair_counts:
                continue

            word_weight = word_weights[word_id]
            remove_old_pair_counts(
                old_pair_counts=old_pair_counts,
                word_id=word_id,
                word_weight=word_weight,
                pair_counts=pair_counts,
                pair_to_word_counts=pair_to_word_counts,
                heap=heap,
                vocab=vocab,
            )

            new_token_ids = merge_token_id_pair(old_token_ids, best_pair, merged_token_id)
            word_tokens[word_id] = new_token_ids
            add_new_pair_counts(
                new_pair_counts=count_token_id_pairs(new_token_ids),
                word_id=word_id,
                word_weight=word_weight,
                pair_counts=pair_counts,
                pair_to_word_counts=pair_to_word_counts,
                heap=heap,
                vocab=vocab,
            )

    return vocab, merges


def count_token_id_pairs(token_ids: Sequence[int]) -> dict[IdPair, int]:
    pair_counts: dict[IdPair, int] = {}
    for index in range(len(token_ids) - 1):
        pair = (token_ids[index], token_ids[index + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def remove_old_pair_counts(
    old_pair_counts: Mapping[IdPair, int],
    word_id: int,
    word_weight: int,
    pair_counts: dict[IdPair, int],
    pair_to_word_counts: dict[IdPair, dict[int, int]],
    heap: list[tuple[int, DescendingPairKey, IdPair]],
    vocab: Mapping[int, bytes],
) -> None:
    """从全局 pair 统计中移除一个 pretoken 合并前贡献的 pair。"""
    for pair, occurrences in old_pair_counts.items():
        weighted_count = occurrences * word_weight
        updated_count = pair_counts.get(pair, 0) - weighted_count
        if updated_count > 0:
            pair_counts[pair] = updated_count
            push_pair(heap, pair, updated_count, vocab)
        else:
            pair_counts.pop(pair, None)

        word_map = pair_to_word_counts.get(pair)
        if word_map is not None:
            word_map.pop(word_id, None)
            if not word_map:
                pair_to_word_counts.pop(pair, None)


def add_new_pair_counts(
    new_pair_counts: Mapping[IdPair, int],
    word_id: int,
    word_weight: int,
    pair_counts: dict[IdPair, int],
    pair_to_word_counts: dict[IdPair, dict[int, int]],
    heap: list[tuple[int, DescendingPairKey, IdPair]],
    vocab: Mapping[int, bytes],
) -> None:
    """向全局 pair 统计加入一个 pretoken 合并后产生的 pair。"""
    for pair, occurrences in new_pair_counts.items():
        weighted_count = occurrences * word_weight
        updated_count = pair_counts.get(pair, 0) + weighted_count
        pair_counts[pair] = updated_count
        pair_to_word_counts.setdefault(pair, {})[word_id] = occurrences
        push_pair(heap, pair, updated_count, vocab)


def merge_token_id_pair(token_ids: Sequence[int], pair_to_merge: IdPair, merged_token_id: int) -> list[int]:
    """按 BPE 规则从左到右合并非重叠 pair。"""
    merged_token_ids: list[int] = []
    index = 0
    token_count = len(token_ids)

    while index < token_count:
        if index < token_count - 1 and (token_ids[index], token_ids[index + 1]) == pair_to_merge:
            merged_token_ids.append(merged_token_id)
            index += 2
        else:
            merged_token_ids.append(token_ids[index])
            index += 1

    return merged_token_ids


def push_pair(
    heap: list[tuple[int, DescendingPairKey, IdPair]],
    pair: IdPair,
    count: int,
    vocab: Mapping[int, bytes],
) -> None:
    if count <= 0:
        return
    pair_bytes = (vocab[pair[0]], vocab[pair[1]])
    heapq.heappush(heap, (-count, DescendingPairKey(pair_bytes), pair))


def pop_best_pair(
    heap: list[tuple[int, DescendingPairKey, IdPair]],
    pair_counts: Mapping[IdPair, int],
    vocab: Mapping[int, bytes],
) -> IdPair | None:
    while heap:
        negative_count, pair_key, pair = heapq.heappop(heap)
        count = -negative_count
        current_count = pair_counts.get(pair, 0)
        if current_count != count:
            continue
        if pair_key.pair != (vocab[pair[0]], vocab[pair[1]]):
            continue
        return pair
    return None


# ===== TinyStories 修改点 4：训练任务编排、序列化、资源统计和最长 token 分析。 =====


def serialize_training_artifacts(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[Pair],
    summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    summary_path = output_dir / "summary.json"

    vocab_payload = {
        str(token_id): {
            "hex": token_bytes.hex(),
            "utf8": decode_for_inspection(token_bytes),
            "byte_length": len(token_bytes),
        }
        for token_id, token_bytes in vocab.items()
    }
    merges_payload = [
        {
            "rank": rank,
            "left_hex": left.hex(),
            "right_hex": right.hex(),
            "left_utf8": decode_for_inspection(left),
            "right_utf8": decode_for_inspection(right),
        }
        for rank, (left, right) in enumerate(merges)
    ]

    vocab_path.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    merges_path.write_text(json.dumps(merges_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"vocab": vocab_path, "merges": merges_path, "summary": summary_path}


def decode_for_inspection(token_bytes: bytes) -> str:
    return token_bytes.decode("utf-8", errors="replace")


def find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    token_id, token_bytes = max(vocab.items(), key=lambda item: (len(item[1]), item[0]))
    return token_id, token_bytes


def train_tinystories_bpe(
    input_path: Path,
    output_dir: Path,
    vocab_size: int,
    special_tokens: Sequence[str],
    workers: int,
    batch_bytes: int,
) -> tuple[dict[int, bytes], list[Pair], TrainingMetrics, dict[str, Path]]:
    normalized_special_tokens = validate_train_bpe_inputs(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
    )

    total_start = time.perf_counter()
    with ProcessTreeMemorySampler() as memory_sampler:
        pretokenization_start = time.perf_counter()
        word_counts = count_pretokens_parallel(
            input_path=input_path,
            special_tokens=normalized_special_tokens,
            workers=workers,
            batch_bytes=batch_bytes,
        )
        pretokenization_seconds = time.perf_counter() - pretokenization_start

        bpe_training_start = time.perf_counter()
        vocab, merges = train_bpe_from_pretoken_counts(
            word_counts=word_counts,
            vocab_size=vocab_size,
            special_tokens=normalized_special_tokens,
        )
        bpe_training_seconds = time.perf_counter() - bpe_training_start

        longest_token_id, longest_token = find_longest_token(vocab)
        serialization_start = time.perf_counter()
        summary = {
            "input_path": str(input_path),
            "vocab_size": vocab_size,
            "special_tokens": normalized_special_tokens,
            "workers": workers,
            "batch_bytes": batch_bytes,
            "unique_pretokens": len(word_counts),
            "total_pretokens": sum(word_counts.values()),
            "actual_vocab_size": len(vocab),
            "merge_count": len(merges),
            "pretokenization_seconds": pretokenization_seconds,
            "bpe_training_seconds": bpe_training_seconds,
            "longest_token": {
                "id": longest_token_id,
                "hex": longest_token.hex(),
                "utf8": decode_for_inspection(longest_token),
                "byte_length": len(longest_token),
            },
        }
        artifact_paths = serialize_training_artifacts(
            output_dir=output_dir,
            vocab=vocab,
            merges=merges,
            summary=summary,
        )
        serialization_seconds = time.perf_counter() - serialization_start

    total_seconds = time.perf_counter() - total_start
    metrics = TrainingMetrics(
        total_seconds=total_seconds,
        pretokenization_seconds=pretokenization_seconds,
        bpe_training_seconds=bpe_training_seconds,
        serialization_seconds=serialization_seconds,
        peak_rss_mb=memory_sampler.peak_rss_mb,
        unique_pretokens=len(word_counts),
        total_pretokens=sum(word_counts.values()),
    )

    final_summary = {
        **summary,
        "total_seconds": total_seconds,
        "serialization_seconds": serialization_seconds,
        "peak_rss_mb": memory_sampler.peak_rss_mb,
    }
    artifact_paths["summary"].write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return vocab, merges, metrics, artifact_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 10k byte-level BPE tokenizer on TinyStories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="TinyStories training text path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON artifacts.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Maximum vocabulary size.")
    parser.add_argument(
        "--special-token",
        action="append",
        default=[DEFAULT_SPECIAL_TOKEN],
        help="Special token to add. May be passed multiple times.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Pretokenizer workers.")
    parser.add_argument("--batch-bytes", type=int, default=DEFAULT_BATCH_BYTES, help="Bytes per pretokenization batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab, merges, metrics, artifact_paths = train_tinystories_bpe(
        input_path=args.input,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        workers=args.workers,
        batch_bytes=args.batch_bytes,
    )
    longest_token_id, longest_token = find_longest_token(vocab)

    print(f"vocab_size={len(vocab)}")
    print(f"merges={len(merges)}")
    print(f"total_seconds={metrics.total_seconds:.3f}")
    print(f"pretokenization_seconds={metrics.pretokenization_seconds:.3f}")
    print(f"bpe_training_seconds={metrics.bpe_training_seconds:.3f}")
    print(f"serialization_seconds={metrics.serialization_seconds:.3f}")
    print(f"peak_rss_mb={metrics.peak_rss_mb:.1f}")
    print(f"unique_pretokens={metrics.unique_pretokens}")
    print(f"total_pretokens={metrics.total_pretokens}")
    print(
        "longest_token="
        f"id={longest_token_id} "
        f"bytes={len(longest_token)} "
        f"utf8={decode_for_inspection(longest_token)!r} "
        f"hex={longest_token.hex()}"
    )
    print(f"vocab_path={artifact_paths['vocab']}")
    print(f"merges_path={artifact_paths['merges']}")
    print(f"summary_path={artifact_paths['summary']}")


if __name__ == "__main__":
    main()
