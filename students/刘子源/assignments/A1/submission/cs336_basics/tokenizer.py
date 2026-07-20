# 训练和编码全挤在这一个文件里了，历史包袱，先别拆，不然作业脚本的导入又得跟着改。

from __future__ import annotations

import base64
import heapq
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import regex


# GPT-2 那条祖传正则。看不顺眼也别格式化，少个空格都可能对不上官方结果。
GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 小文件开进程属于花钱买慢，64M 以下老实单进程。
DEFAULT_PARALLEL_THRESHOLD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_WORKERS = 4


def _split_ordinary_text(text: str, special_tokens: list[str]) -> Iterator[str]:
    special_list = sorted(set(special_tokens), key=len, reverse=True)
    if not special_list:
        yield text
        return
    cut_re = regex.compile("(" + "|".join(regex.escape(token) for token in special_list) + ")")
    for part in cut_re.split(text):
        if part and part not in special_list:
            yield part


class _MaxHeapEntry:
    __slots__ = ("count", "pair", "pair_bytes")

    def __init__(self, count: int, pair: tuple[int, int], pair_bytes: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair
        self.pair_bytes = pair_bytes

    def __lt__(self, other: _MaxHeapEntry) -> bool:
        # heapq 只会弹最小的，这里反着比，硬把它当最大堆使。
        if self.count != other.count:
            return self.count > other.count
        return self.pair_bytes > other.pair_bytes


def _find_chunk_boundaries(
    input_path: str | Path,
    desired_num_chunks: int,
    special_tokens: list[str],
    scan_block_size: int = 1024 * 1024,
) -> list[int]:
    path = Path(input_path)
    file_size = path.stat().st_size
    if desired_num_chunks <= 1 or file_size == 0 or not special_tokens:
        return [0, file_size]

    special_byte_list = [token.encode("utf-8") for token in special_tokens]
    max_special_len = max(map(len, special_byte_list))
    guess_size = max(1, file_size // desired_num_chunks)
    cut_list = [0]

    with path.open("rb") as input_file:
        for chunk_i in range(1, desired_num_chunks):
            guess_pos = chunk_i * guess_size
            if guess_pos >= file_size:
                break
            input_file.seek(guess_pos)
            abs_pos = guess_pos
            tail = b""
            cut_pos = file_size

            while abs_pos < file_size:
                block = input_file.read(scan_block_size)
                if not block:
                    break
                can_search = tail + block
                search_start = abs_pos - len(tail)
                found_list = [can_search.find(token) for token in special_byte_list]
                found_list = [offset for offset in found_list if offset >= 0]
                if found_list:
                    cut_pos = search_start + min(found_list)
                    break
                tail_len = min(max_special_len - 1, len(can_search))
                tail = can_search[-tail_len:] if tail_len else b""
                abs_pos += len(block)

            cut_list.append(cut_pos)

    cut_list.append(file_size)
    return sorted(set(cut_list))


def _read_utf8_range(
    input_path: str | Path,
    start: int,
    end: int,
    *,
    range_label: str = "file",
) -> str:
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        raw = input_file.read(end - start)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise UnicodeDecodeError(
            error.encoding,
            error.object,
            error.start,
            error.end,
            f"invalid UTF-8 in {range_label} range [{start}, {end}): {error.reason}",
        ) from error
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _count_pretokens_in_chunk(task: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    input_path, start, end, special_tokens = task
    chunk = _read_utf8_range(input_path, start, end)

    word_count: Counter[bytes] = Counter()
    for normal_text in _split_ordinary_text(chunk, list(special_tokens)):
        for match in regex.finditer(GPT2_PRETOKEN_PATTERN, normal_text):
            word_count[match.group().encode("utf-8")] += 1
    return word_count


def _resolve_num_workers(
    input_path: str | Path,
    requested_workers: int | None,
    special_tokens: list[str],
    parallel_threshold_bytes: int,
) -> int:
    if requested_workers is not None and requested_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if requested_workers is not None:
        return requested_workers
    if not special_tokens or Path(input_path).stat().st_size < parallel_threshold_bytes:
        return 1
    slurm_cpu = os.environ.get("SLURM_CPUS_PER_TASK")
    can_use_cpu = int(slurm_cpu) if slurm_cpu else (os.cpu_count() or 1)
    return max(1, min(DEFAULT_MAX_WORKERS, can_use_cpu))


def _count_pretokens(
    input_path: str | Path,
    special_tokens: list[str],
    num_workers: int,
    chunks_per_worker: int,
    verbose: bool,
) -> Counter[bytes]:
    want_chunks = max(1, num_workers * chunks_per_worker)
    cut_list = _find_chunk_boundaries(input_path, want_chunks, special_tokens)
    jobs = [
        (str(input_path), start, end, tuple(special_tokens))
        for start, end in zip(cut_list[:-1], cut_list[1:])
        if end > start
    ]
    real_workers = min(num_workers, len(jobs))
    if verbose:
        print(
            f"[BPE] pre-tokenization: {len(jobs)} chunks, {real_workers} worker(s)",
            file=sys.stderr,
            flush=True,
        )
        if real_workers < num_workers:
            print(
                "[BPE] fewer safe chunk boundaries were found; falling back to fewer workers",
                file=sys.stderr,
                flush=True,
            )

    all_count: Counter[bytes] = Counter()
    if real_workers <= 1:
        for job in jobs:
            all_count.update(_count_pretokens_in_chunk(job))
        return all_count

    # 别一股脑 submit，Future 会抱着一堆大 Counter 不撒手，内存分分钟炸。
    with ProcessPoolExecutor(max_workers=real_workers) as pool:
        job_iter = iter(jobs)
        running = {}
        for _ in range(real_workers):
            job = next(job_iter, None)
            if job is not None:
                running[pool.submit(_count_pretokens_in_chunk, job)] = job

        done_num = 0
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                all_count.update(future.result())
                done_num += 1
                if verbose:
                    print(
                        f"[BPE] pre-tokenization chunks: {done_num}/{len(jobs)}",
                        file=sys.stderr,
                        flush=True,
                    )
                next_job = next(job_iter, None)
                if next_job is not None:
                    running[pool.submit(_count_pretokens_in_chunk, next_job)] = next_job
    return all_count


def _merge_token_pair(tokens: list[int], pair: tuple[int, int], merged_id: int) -> list[int]:
    merged_list: list[int] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged_list.append(merged_id)
            i += 2
        else:
            merged_list.append(tokens[i])
            i += 1
    return merged_list


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int | None = None,
    chunks_per_worker: int = 4,
    parallel_threshold_bytes: int = DEFAULT_PARALLEL_THRESHOLD_BYTES,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    chunks_per_worker = int(chunks_per_worker)
    parallel_threshold_bytes = int(parallel_threshold_bytes)
    if chunks_per_worker < 1:
        raise ValueError("chunks_per_worker must be at least 1")
    if parallel_threshold_bytes < 0:
        raise ValueError("parallel_threshold_bytes must be non-negative")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must be unique")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size cannot contain the required base vocabulary")

    # 头 256 个位置一字节一个，别问，byte BPE 的地基就是这么朴素。
    vocabulary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocabulary[len(vocabulary)] = special_token.encode("utf-8")

    resolved_workers = _resolve_num_workers(
        input_path, num_workers, special_tokens, parallel_threshold_bytes
    )
    # key 用 bytes 省点内存，OWT 那体量不省真扛不住。
    pretoken_counts = _count_pretokens(
        input_path,
        special_tokens,
        resolved_workers,
        chunks_per_worker,
        verbose,
    )

    # words 会被一轮轮改，下面那些倒排表就是为了别每次重扫全语料。
    words = [list(word) for word in pretoken_counts]
    frequencies = [pretoken_counts[word] for word in pretoken_counts]
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for word_index, (tokens, frequency) in enumerate(zip(words, frequencies)):
        adjacent_pairs = Counter(zip(tokens, tokens[1:]))
        for pair, occurrences in adjacent_pairs.items():
            pair_counts[pair] += occurrences * frequency
            pair_to_words[pair].add(word_index)

    max_heap: list[_MaxHeapEntry] = []

    def pair_bytes(pair: tuple[int, int]) -> tuple[bytes, bytes]:
        return vocabulary[pair[0]], vocabulary[pair[1]]

    for pair, count in pair_counts.items():
        heapq.heappush(max_heap, _MaxHeapEntry(count, pair, pair_bytes(pair)))

    merges: list[tuple[bytes, bytes]] = []
    target_num_merges = vocab_size - len(vocabulary)
    while len(vocabulary) < vocab_size:
        # 堆里旧账懒得当场删，弹出来发现过期再扔，便宜点。
        while max_heap:
            entry = heapq.heappop(max_heap)
            if pair_counts.get(entry.pair, 0) == entry.count and entry.count > 0:
                selected_pair = entry.pair
                break
        else:
            break

        selected_bytes = pair_bytes(selected_pair)
        merges.append(selected_bytes)
        merged_id = len(vocabulary)
        vocabulary[merged_id] = selected_bytes[0] + selected_bytes[1]
        affected_words = list(pair_to_words.pop(selected_pair, set()))
        changed_pairs: set[tuple[int, int]] = set()

        for word_index in affected_words:
            old_tokens = words[word_index]
            old_pairs = Counter(zip(old_tokens, old_tokens[1:]))
            frequency = frequencies[word_index]
            # 先把旧账扣掉，再把 merge 后的新账加回来。
            for pair, occurrences in old_pairs.items():
                pair_counts[pair] -= occurrences * frequency
                word_indexes = pair_to_words.get(pair)
                if word_indexes is not None:
                    word_indexes.discard(word_index)
                    if not word_indexes:
                        pair_to_words.pop(pair, None)
                changed_pairs.add(pair)

            new_tokens = _merge_token_pair(old_tokens, selected_pair, merged_id)
            words[word_index] = new_tokens
            new_pairs = Counter(zip(new_tokens, new_tokens[1:]))
            for pair, occurrences in new_pairs.items():
                pair_counts[pair] += occurrences * frequency
                pair_to_words[pair].add(word_index)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(max_heap, _MaxHeapEntry(count, pair, pair_bytes(pair)))
            else:
                # 死 key 留着只会越攒越多，顺手清掉。
                pair_counts.pop(pair, None)

        # 懒删除攒太多垃圾后就整锅重建，不然堆最后胖得不像话。
        active_pair_count = len(pair_to_words)
        if len(max_heap) > max(10_000, 4 * active_pair_count):
            max_heap = [
                _MaxHeapEntry(pair_counts[pair], pair, pair_bytes(pair))
                for pair in pair_to_words
                if pair_counts.get(pair, 0) > 0
            ]
            heapq.heapify(max_heap)
            if verbose:
                print(
                    f"[BPE] compacted max-heap: {len(max_heap)} active entries",
                    file=sys.stderr,
                    flush=True,
                )

        if verbose and (len(merges) % 100 == 0 or len(merges) == target_num_merges):
            print(
                f"[BPE] merges: {len(merges)}/{target_num_merges}; "
                f"active_pairs={len(pair_to_words)} heap_entries={len(max_heap)}",
                file=sys.stderr,
                flush=True,
            )

    return vocabulary, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        # 数字越小资格越老，编码时先合它。
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        if len(self.bytes_to_id) != len(self.vocab):
            raise ValueError("vocabulary byte strings must be unique")
        self.special_tokens = list(special_tokens or [])
        # 外面传来的 special 可能漏在 vocab 里，只能现场补票。
        for special_token in self.special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in self.bytes_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = encoded
                self.bytes_to_id[encoded] = token_id
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        self.special_pattern = (
            regex.compile("(" + "|".join(regex.escape(token) for token in sorted_specials) + ")")
            if sorted_specials
            else None
        )
        self.streaming_lookbehind = max((len(token) for token in sorted_specials), default=1) - 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with Path(vocab_filepath).open(encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)
        with Path(merges_filepath).open(encoding="utf-8") as merges_file:
            raw_merges = json.load(merges_file)
        vocab = {int(token_id): base64.b64decode(token_bytes) for token_id, token_bytes in raw_vocab.items()}
        merges = [(base64.b64decode(left), base64.b64decode(right)) for left, right in raw_merges]
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath, merges_filepath) -> None:
        # JSON 不认识 bytes，只能先套层 base64，土是土但跨机器省事。
        vocab_payload = {
            str(token_id): base64.b64encode(token_bytes).decode("ascii")
            for token_id, token_bytes in self.vocab.items()
        }
        merges_payload = [
            [base64.b64encode(left).decode("ascii"), base64.b64encode(right).decode("ascii")]
            for left, right in self.merges
        ]
        Path(vocab_filepath).write_text(json.dumps(vocab_payload, ensure_ascii=True), encoding="utf-8")
        Path(merges_filepath).write_text(json.dumps(merges_payload, ensure_ascii=True), encoding="utf-8")

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        # 刚进来是一字节一个，后面按老 merge 表慢慢粘。
        symbols = [bytes([byte]) for byte in pretoken.encode("utf-8")]
        while len(symbols) > 1:
            # 直着找最老的 pair。生成器那版挺秀，接手的人看半天，没必要。
            best_rank = None
            best_pair = None
            for pair in zip(symbols, symbols[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged: list[bytes] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == best_pair:
                    merged.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return [self.bytes_to_id[symbol] for symbol in symbols]

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in regex.finditer(GPT2_PRETOKEN_PATTERN, text):
            yield from self._encode_pretoken(match.group())

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return list(self._encode_ordinary(text))
        token_ids: list[int] = []
        # special 得先捞出来，不然那条祖传正则会把它当普通标点嚼了。
        for part in self.special_pattern.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                token_ids.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                token_ids.extend(self._encode_ordinary(part))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending = ""
        for chunk in iterable:
            pending += chunk
            # 尾巴先扣着，special token 可能刚好被 chunk 从中间劈开。
            safe_end = max(0, len(pending) - self.streaming_lookbehind)
            special_matches = list(self.special_pattern.finditer(pending)) if self.special_pattern else []
            consumed = 0
            cursor = 0
            blocked = False
            for special_match in special_matches + [None]:
                ordinary_end = special_match.start() if special_match is not None else len(pending)
                ordinary_matches = list(regex.finditer(GPT2_PRETOKEN_PATTERN, pending[cursor:ordinary_end]))
                for index, match in enumerate(ordinary_matches):
                    start = cursor + match.start()
                    end = cursor + match.end()
                    # "hel" + "lo" 这种破切法真会来，所以最后一个词也得等等。
                    is_trailing_match = (
                        special_match is None
                        and index == len(ordinary_matches) - 1
                        and end == len(pending)
                    )
                    if end > safe_end or is_trailing_match:
                        consumed = start
                        blocked = True
                        break
                    yield from self._encode_pretoken(match.group())
                    consumed = end
                if blocked:
                    break
                if special_match is None:
                    break
                if special_match.end() > safe_end:
                    # 太贴尾巴就先别认，没准下一块来了个更长的重叠 special。
                    consumed = special_match.start()
                    break
                yield self.bytes_to_id[special_match.group().encode("utf-8")]
                consumed = special_match.end()
                cursor = special_match.end()
            pending = pending[consumed:]
        if pending:
            yield from self.encode(pending)

    def decode(self, ids: list[int]) -> str:
        # 得先拼 bytes 再 decode，逐 token 解中文会把一个字拆烂。
        try:
            raw_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID: {error.args[0]}") from error
        return raw_bytes.decode("utf-8", errors="replace")
