"""字节对编码（BPE）分词器相关工具。"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator

import regex as re


GPT2_PRETOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _count_pretokens(
    text: str,
    special_tokens: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> Counter[bytes]:
    """统计 GPT-2 预分词结果，并排除文本中出现的特殊 token。"""
    counts: Counter[bytes] = Counter()

    def count_chunk(chunk: str) -> None:
        counts.update(match.group().encode("utf-8") for match in GPT2_PRETOKEN_PATTERN.finditer(chunk))

    if special_tokens:
        # 优先匹配更长的特殊 token，确保特殊 token 相互重叠时结果仍然确定。
        special_pattern = "|".join(
            re.escape(token) for token in sorted(set(special_tokens), key=lambda token: (-len(token), token))
        )
        last_end = 0
        for special_match in re.finditer(special_pattern, text):
            count_chunk(text[last_end : special_match.start()])
            last_end = special_match.end()
            if progress_callback:
                progress_callback(last_end, len(text))
        count_chunk(text[last_end:])
    else:
        count_chunk(text)
    if progress_callback:
        progress_callback(len(text), len(text))
    return counts


def _find_chunk_boundaries(
    input_path: str | os.PathLike[str], num_processes: int, split_token: bytes
) -> list[int]:
    """在特殊 token 起始位置切分文件，保证每个分块可独立预分词。"""
    file_size = os.path.getsize(input_path)
    if num_processes == 1 or not split_token:
        return [0, file_size]

    boundaries = [0]
    search_block_size = 1 << 20
    with open(input_path, "rb") as corpus:
        for chunk_index in range(1, num_processes):
            position = file_size * chunk_index // num_processes
            corpus.seek(position)
            while position < file_size:
                block = corpus.read(search_block_size)
                if not block:
                    position = file_size
                    break
                found_at = block.find(split_token)
                if found_at >= 0:
                    position += found_at
                    break
                position += len(block)
            boundaries.append(position)
    boundaries.append(file_size)
    return sorted(set(boundaries))


def _count_pretokens_in_file_chunk(
    input_path: str | os.PathLike[str], start: int, end: int, special_tokens: list[str]
) -> tuple[int, Counter[bytes]]:
    """供子进程调用：读取一个安全分块并统计其中的预分词。"""
    with open(input_path, "rb") as corpus:
        corpus.seek(start)
        chunk = corpus.read(end - start).decode("utf-8")
    return end - start, _count_pretokens(chunk, special_tokens)


def _count_pretokens_parallel(
    input_path: str | os.PathLike[str],
    special_tokens: list[str],
    num_processes: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Counter[bytes]:
    """按特殊 token 切分文件，并行统计每块的 GPT-2 预分词频次。"""
    split_token = max((token.encode("utf-8") for token in special_tokens), key=len, default=b"")
    boundaries = _find_chunk_boundaries(input_path, num_processes, split_token)
    chunks = list(zip(boundaries, boundaries[1:]))
    total_bytes = os.path.getsize(input_path)
    counts: Counter[bytes] = Counter()
    processed_bytes = 0

    with ProcessPoolExecutor(max_workers=min(num_processes, len(chunks))) as executor:
        futures = [
            executor.submit(_count_pretokens_in_file_chunk, input_path, start, end, special_tokens)
            for start, end in chunks
            if start != end
        ]
        for future in as_completed(futures):
            chunk_size, chunk_counts = future.result()
            counts.update(chunk_counts)
            processed_bytes += chunk_size
            if progress_callback:
                progress_callback(processed_bytes, total_bytes)
    return counts


class Tokenizer:
    """使用给定词表和合并规则进行编码、解码的字节级 BPE 分词器。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.special_token_ids = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }
        if len(self.special_token_ids) != len(self.special_tokens):
            raise ValueError("每个特殊 token 都必须存在于词表中")

        if self.special_tokens:
            # 捕获分组使 split 的结果保留特殊 token，以便直接转换为其 token ID。
            pattern = "|".join(
                re.escape(token) for token in sorted(self.special_tokens, key=lambda token: (-len(token), token))
            )
            self.special_pattern: re.Pattern | None = re.compile(f"({pattern})")
        else:
            self.special_pattern = None

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """对一个预分词执行按 merge rank 优先的 BPE 合并。"""
        symbols = [bytes([byte]) for byte in pretoken.encode("utf-8")]
        while len(symbols) > 1:
            candidates = [
                (self.merge_ranks[pair], index)
                for index, pair in enumerate(zip(symbols, symbols[1:]))
                if pair in self.merge_ranks
            ]
            if not candidates:
                break
            _, index = min(candidates)
            symbols[index : index + 2] = [symbols[index] + symbols[index + 1]]
        return [self.token_to_id[symbol] for symbol in symbols]

    def _encode_text(self, text: str) -> Iterator[int]:
        """编码不包含特殊 token 的普通文本。"""
        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group())

    def encode(self, text: str) -> list[int]:
        """将文本编码为 token ID 列表，已声明的特殊 token 保持为单个 token。"""
        return list(self.iter_encode(text))

    def iter_encode(self, text: str) -> Iterator[int]:
        """惰性编码单个字符串，已声明的特殊 token 保持为单个 token。"""
        if not self.special_pattern:
            yield from self._encode_text(text)
            return

        for chunk in self.special_pattern.splititer(text):
            if chunk in self.special_token_ids:
                yield self.special_token_ids[chunk]
            else:
                yield from self._encode_text(chunk)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐项编码文本迭代器，避免一次读取整个输入。"""
        for text in iterable:
            yield from self.iter_encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        """将 token ID 序列还原为 UTF-8 文本。"""
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


def train_bpe_tokenizer(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    progress: bool = False,
    progress_interval: int = 100,
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """使用 ``input_path`` 指向的语料训练字节级 BPE 分词器。

    返回的词表依次包含全部 256 个单字节、特殊 token，以及训练得到的合并
    token。当多个字节对出现次数相同时，按照作业要求选择字典序最大的字节对。
    """
    unique_special_tokens = list(dict.fromkeys(special_tokens))
    if num_processes < 1:
        raise ValueError("num_processes 必须至少为 1")
    minimum_size = 256 + len(unique_special_tokens)
    if vocab_size < minimum_size:
        raise ValueError(f"vocab_size must be at least {minimum_size}")

    vocab = {byte: bytes([byte]) for byte in range(256)}
    for token in unique_special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if progress:
        print("正在读取语料并进行预分词统计...", flush=True)

    next_progress_percentage = 1

    def report_pretoken_progress(processed: int, total: int) -> None:
        nonlocal next_progress_percentage
        percentage = 100 * processed / total if total else 100
        if percentage >= next_progress_percentage or processed == total:
            print(f"预分词进度：{percentage:.0f}% ({processed:,}/{total:,} 已处理)。", flush=True)
            next_progress_percentage = int(percentage) + 1

    input_path = os.fspath(input_path)
    if num_processes == 1:
        with open(input_path, encoding="utf-8") as corpus:
            corpus_text = corpus.read()
        pretoken_counts = _count_pretokens(
            corpus_text,
            unique_special_tokens,
            report_pretoken_progress if progress else None,
        )
    else:
        pretoken_counts = _count_pretokens_parallel(
            input_path,
            unique_special_tokens,
            num_processes,
            report_pretoken_progress if progress else None,
        )

    if progress:
        print(f"预分词完成：共 {len(pretoken_counts):,} 个不同预分词。", flush=True)

    # 将每个不同的预分词表示为当前 BPE 符号组成的元组。
    words = {
        word_id: tuple(bytes([byte]) for byte in pretoken)
        for word_id, pretoken in enumerate(pretoken_counts)
    }
    frequencies = dict(enumerate(pretoken_counts.values()))
    merges: list[tuple[bytes, bytes]] = []

    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, word in words.items():
        frequency = frequencies[word_id]
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency
            pair_words[pair].add(word_id)

    if progress:
        print(f"开始 BPE 合并：词表 {len(vocab):,} → {vocab_size:,}。", flush=True)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        pair = max(pair_counts, key=lambda candidate: (pair_counts[candidate], candidate))
        merged_symbol = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = merged_symbol

        # 只有包含当前待合并字节对的词才会发生变化。先从全局统计中移除这些词
        # 原有的字节对贡献，完成合并后，再加入新的字节对贡献。
        affected_word_ids = tuple(pair_words[pair])
        for word_id in affected_word_ids:
            word = words[word_id]
            frequency = frequencies[word_id]
            for old_pair in zip(word, word[1:]):
                pair_counts[old_pair] -= frequency
                pair_words[old_pair].discard(word_id)
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                    del pair_words[old_pair]

            merged_word: list[bytes] = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
                    merged_word.append(merged_symbol)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1
            words[word_id] = tuple(merged_word)

            for new_pair in zip(merged_word, merged_word[1:]):
                pair_counts[new_pair] += frequency
                pair_words[new_pair].add(word_id)

        completed_merges = len(merges)
        if progress and (completed_merges % progress_interval == 0 or len(vocab) == vocab_size):
            percentage = 100 * len(vocab) / vocab_size
            print(
                f"已完成 {completed_merges:,} 次合并；词表大小 {len(vocab):,}/{vocab_size:,} "
                f"({percentage:.1f}%)。",
                flush=True,
            )

    if progress:
        print(f"训练完成：生成 {len(merges):,} 条 merge 规则。", flush=True)

    return vocab, merges


train_bpe = train_bpe_tokenizer
