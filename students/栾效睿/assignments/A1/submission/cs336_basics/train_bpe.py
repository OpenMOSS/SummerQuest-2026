import heapq
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threading import Lock
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
heap = []
word_count = Counter()
record_all: dict[tuple[bytes, bytes], int] = {}
record_key: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)


def iter_special_token_batches(
    in_path: Path,
    special_token: str,
    target_batch_bytes: int,
) -> Iterator[bytes]:
    """产出以 special token 为硬边界的 byte batch，避免 merge 或预分词跨文档。"""
    delimiter = special_token.encode("utf-8")
    read_size = min(target_batch_bytes, 16 * 1024 * 1024)
    remainder = b""
    batch = bytearray()

    with in_path.open("rb") as input_file:
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


merge_lock = Lock()  # 确保多线程回调时合并安全


def merge_result(future):
    """回调函数：每个进程处理完一个 64MB batch 后自动调用"""
    local_res = future.result()
    with merge_lock:
        word_count.update(local_res)


# 2. 主逻辑：流式提交
def run_pipeline(in_path: Path, tokens: list[str], target_batch_bytes=64 * 1024 * 1024, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 使用生成器直接迭代，不产生巨大的任务列表
        for batch in iter_special_token_batches(in_path, tokens[0], target_batch_bytes):
            # 提交任务
            future = executor.submit(pre_tokenization, (batch, tokens))
            # 绑定回调，处理完一个就合一个
            future.add_done_callback(merge_result)

    return word_count


def pre_tokenization(args) -> dict[str, int]:
    bytes_batch, tokens = args

    input_str = bytes_batch.decode("utf-8")
    if tokens is not None:
        pattern = "|".join(re.escape(token) for token in tokens)
        parts = re.split(f"({pattern})", input_str)
    else:
        parts = [input_str]

    # print(parts)
    res: dict[str, int] = {}
    for part in parts:
        if part not in tokens:
            for match in re.finditer(PAT, part):
                word = match.group(0)
                res[word] = res.get(word, 0) + 1
    return res


class HeapItem:
    def __init__(self, cnt, byte_pair):
        self.cnt = cnt
        self.byte_pair = byte_pair

    def __lt__(self, other):
        # 核心逻辑：
        # 1. 如果 cnt 不相等，cnt 大的优先 (所以 return self.cnt > other.cnt)
        if self.cnt != other.cnt:
            return self.cnt > other.cnt
        # 2. 如果 cnt 相等，byte_pair 大的优先
        return self.byte_pair > other.byte_pair

    def __repr__(self):
        return f"({self.cnt}, {self.byte_pair})"


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, string in enumerate(special_tokens):
        vocab[i + 256] = string.encode("utf-8")
    return vocab


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_path: Path,
) -> dict[str, Path]:
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_path = output_path / "vocab.json"
    merges_path = output_path / "merges.json"

    vocab_payload = {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()}
    merges_payload = [[left.hex(), right.hex()] for left, right in merges]

    vocab_path.write_text(json.dumps(vocab_payload, indent=2), encoding="utf-8")
    merges_path.write_text(json.dumps(merges_payload, indent=2), encoding="utf-8")

    return {"vocab": vocab_path, "merges": merges_path}


def init_heap() -> None:
    heap.clear()
    for word, cnt in word_count.items():
        word_bytes = tuple([bytes([byte]) for byte in word.encode("utf-8")])
        for bp in zip(word_bytes[0:], word_bytes[1:]):
            record_all[bp] = record_all.get(bp, 0) + cnt
            record_key[bp].add(word_bytes)
    for bp, cnt in record_all.items():
        heap_item = HeapItem(cnt, bp)
        heapq.heappush(heap, heap_item)


def get_one_byte_pair() -> tuple[bytes, bytes] | None:
    target_bytes_pair: tuple[bytes, bytes] | None = None
    while heap:
        heap_item: HeapItem = heapq.heappop(heap)
        target_bytes_pair: tuple[bytes, bytes] = heap_item.byte_pair
        if record_all[target_bytes_pair] == heap_item.cnt:
            break
        else:
            count = record_all[target_bytes_pair]
            heapq.heappush(
                heap,
                HeapItem(count, target_bytes_pair),
            )
    if target_bytes_pair is not None:
        merge_one_step(target_bytes_pair, tuple_list=list(record_key[target_bytes_pair]))
    return target_bytes_pair


def merge_one_step(byte_pair: tuple[bytes, bytes], tuple_list: list[tuple[bytes, ...]]) -> None:
    new_tuple_list = [modify_key(key=word_bytes, pair_bytes=byte_pair) for word_bytes in tuple_list]

    new_pair_list = set()
    tmp = byte_pair[0] + byte_pair[1]

    # update record_all and record_key
    for word_bytes, new_word_bytes in zip(tuple_list, new_tuple_list):
        word_str = b"".join(word_bytes).decode("utf-8")
        cnt = word_count[word_str]

        # sub all
        for bp in zip(word_bytes[0:], word_bytes[1:]):
            record_all[bp] = record_all.get(bp, 0) - cnt
            record_key[bp].discard(word_bytes)

        # add all
        for bp in zip(new_word_bytes[0:], new_word_bytes[1:]):
            if bp[0] == tmp or bp[1] == tmp:
                new_pair_list.add(bp)
            record_all[bp] = record_all.get(bp, 0) + cnt
            record_key[bp].add(new_word_bytes)

    for new_bp in new_pair_list:
        heapq.heappush(heap, HeapItem(record_all[new_bp], new_bp))


def modify_key(key: tuple[bytes, ...], pair_bytes: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    tmp: list[bytes] = []
    idx = 0
    while idx < len(key) - 1:
        if (key[idx], key[idx + 1]) == pair_bytes:
            tmp.append(key[idx] + key[idx + 1])
            idx += 2
        else:
            tmp.append(key[idx])
            idx += 1
    if idx == len(key) - 1:
        tmp.append(key[idx])
    return tuple(tmp)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # sort special_tokens
    special_tokens.sort(key=lambda x: (-len(x), x))
    # init vocab
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # clear content
    word_count.clear()
    record_all.clear()
    record_key.clear()

    # run_pre_tokenization
    workers = int(kwargs.get("workers") or (os.cpu_count() or 4))
    target_batch_bytes = int(kwargs.get("target_batch_bytes", 64 * 1024 * 1024))
    run_pipeline(
        in_path=Path(input_path),
        tokens=special_tokens,
        target_batch_bytes=target_batch_bytes,
        workers=workers,
    )

    # init_heap
    init_heap()

    # build_the_vocab
    while len(vocab) < vocab_size:
        bytes_pair = get_one_byte_pair()
        if bytes_pair is None:
            break
        merges.append(bytes_pair)
        next_id = len(vocab)
        vocab[next_id] = merges[-1][0] + merges[-1][1]

    # save merges and vocab
    output_dir: str | None = kwargs.get("output_dir")
    if output_dir is not None:
        save_vocab_and_merges(vocab, merges, Path(output_dir))

    return vocab, merges


def main():
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    output_dir = "data/owt_train_bpe"
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000
    # output_dir = "data/TinyStories_train_bpe"
    special_tokens = ["<|endoftext|>"]
    start_time = time.time()

    # cProfile.run("run_train_bpe(input_path, vocab_size, special_tokens)")
    run_train_bpe(input_path, vocab_size, special_tokens, output_dir=output_dir)  # noqa: F821
    end_time = time.time()
    print("cost_time", end_time - start_time)


if __name__ == "__main__":
    main()
