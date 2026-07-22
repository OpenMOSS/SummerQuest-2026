from __future__ import annotations

import base64
import cProfile
import codecs
import heapq
import json
import multiprocessing as mp
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

import regex

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED = regex.compile(GPT2_SPLIT_PATTERN)

class _MaxBytes:
    """Wrap bytes so that lexicographically larger bytes sort smaller.

    Used as a heap key to turn Python's min-heap into a max-heap by bytes
    while preserving CS336's tie-break (count desc, then bytes desc).
    Custom class is needed because bit-flipping via translate() does not
    reverse the prefix relation (shorter bytes still sort before longer
    when one is a prefix of the other).
    """

    __slots__ = ("b",)

    def __init__(self, b: bytes) -> None:
        self.b = b

    def __lt__(self, other: "_MaxBytes") -> bool:
        return self.b > other.b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaxBytes):
            return NotImplemented
        return self.b == other.b

_last_worker_prof_files: list[str] | None = None


def get_last_worker_prof_files() -> list[str] | None:
    """Return prof file paths from the last train_bpe call with profile_workers=True.

    Returns None if profiling was disabled. Files live in a temp dir that the
    caller is responsible for cleaning up (e.g. shutil.rmtree(os.path.dirname(files[0]))).
    """
    return _last_worker_prof_files


def _find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be counted independently.

    Boundaries are aligned to the start of split_special_token so chunks
    can be processed in parallel without pre-token or special-token
    boundary issues.
    """
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
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


def _count_pretokens_range(
    input_path,
    start: int,
    end: int,
    special_tokens: list[str],
    profile: bool = False,
    profile_dir: str | None = None,
) -> Counter[bytes]:
    """Count pre-tokens in byte range [start, end) of input_path.

    The range must start and end at special-token boundaries (or file edges)
    so no pre-token or special token spans the boundary.

    If profile=True, enables cProfile around the counting loop and dumps stats
    to profile_dir/worker_{pid}.prof. The caller collects these files.
    """
    profiler = None
    if profile:
        assert profile_dir is not None, "profile_dir required when profile=True"
        profiler = cProfile.Profile()
        profiler.enable()

    CHUNK_SIZE = 64 * 1024 * 1024  # 64MB sub-chunks
    max_special_len = max((len(t) for t in special_tokens), default=0)
    special_safety = max(0, max_special_len - 1)

    special_pattern = None
    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        special_pattern = re.compile(pattern)

    word_counts_str: Counter[str] = Counter()

    def process_segment(seg: str, is_final: bool) -> str:
        if is_final:
            for tok in _COMPILED.findall(seg):
                word_counts_str[tok] += 1
            return ""
        it = _COMPILED.finditer(seg)
        try:
            prev = next(it)
        except StopIteration:
            return seg
        for m in it:
            word_counts_str[prev.group()] += 1
            prev = m
        return seg[prev.start():]

    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    with open(input_path, "rb") as f:
        f.seek(start)
        remaining = end - start

        while remaining > 0:
            read_size = min(CHUNK_SIZE, remaining)
            raw = f.read(read_size)
            if not raw:
                break
            remaining -= len(raw)
            chunk = decoder.decode(raw)
            buffer += chunk

            if special_pattern:
                special_matches = list(special_pattern.finditer(buffer))
                last_end = 0
                for m in special_matches:
                    process_segment(buffer[last_end:m.start()], is_final=True)
                    last_end = m.end()
                last_seg = buffer[last_end:]
                split_point = max(0, len(last_seg) - special_safety)
                process_part = last_seg[:split_point]
                held_suffix = last_seg[split_point:]
                held_prefix = process_segment(process_part, is_final=False)
                buffer = held_prefix + held_suffix
            else:
                buffer = process_segment(buffer, is_final=False)

        final_chunk = decoder.decode(b"", final=True)
        buffer += final_chunk

        if special_pattern:
            last_end = 0
            for m in special_pattern.finditer(buffer):
                process_segment(buffer[last_end:m.start()], is_final=True)
                last_end = m.end()
            process_segment(buffer[last_end:], is_final=True)
        else:
            process_segment(buffer, is_final=True)

    if profiler:
        profiler.disable()
        path = os.path.join(profile_dir, f"worker_{os.getpid()}.prof")
        profiler.dump_stats(path)

    return Counter({k.encode("utf-8"): v for k, v in word_counts_str.items()})


def _count_pretokens(
    input_path,
    special_tokens: list[str],
    num_processes: int = 4,
    profile_workers: bool = False,
) -> tuple[Counter[bytes], list[str] | None]:
    """Count pre-token frequencies, parallelized across special-token boundaries.

    Splits the file at special-token starts (via _find_chunk_boundaries) so
    each chunk can be counted independently by a worker process. Falls back
    to serial if no special tokens or only one chunk results.

    If profile_workers=True, each worker dumps cProfile stats to a temp dir;
    the returned list contains the .prof file paths (caller reads/cleans up).
    """
    profile_dir = None
    if profile_workers:
        profile_dir = tempfile.mkdtemp(prefix="bpe_profile_")

    file_size = os.path.getsize(input_path)

    if not special_tokens:
        wc = _count_pretokens_range(
            input_path, 0, file_size, special_tokens,
            profile=profile_workers, profile_dir=profile_dir,
        )
        prof_files = _collect_prof_files(profile_dir) if profile_workers else None
        return wc, prof_files

    split_special_token = special_tokens[0].encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processes, split_special_token)

    args = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens,
         profile_workers, profile_dir)
        for i in range(len(boundaries) - 1)
    ]

    if len(args) <= 1:
        results = [_count_pretokens_range(*a) for a in args]
    else:
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(num_processes) as pool:
                results = pool.starmap(_count_pretokens_range, args)
        except Exception:
            results = [_count_pretokens_range(*a) for a in args]

    total_counts: Counter[bytes] = Counter()
    for wc in results:
        total_counts.update(wc)

    prof_files = _collect_prof_files(profile_dir) if profile_workers else None
    return total_counts, prof_files


def _collect_prof_files(profile_dir: str | None) -> list[str] | None:
    """Return sorted .prof file paths in profile_dir, or None if empty."""
    if profile_dir is None:
        return None
    files = sorted(
        os.path.join(profile_dir, f)
        for f in os.listdir(profile_dir)
        if f.endswith(".prof")
    )
    return files if files else None


def train_bpe(
    input_path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int = 4,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    global _last_worker_prof_files
    _last_worker_prof_files = None

    profile_workers = kwargs.pop("profile_workers", False)

    special_tokens = list(special_tokens) if special_tokens else []

    word_counts, prof_files = _count_pretokens(
        input_path, special_tokens, num_processes=num_workers, profile_workers=profile_workers
    )

    if prof_files:
        _last_worker_prof_files = prof_files

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    max_vocab: dict[int, _MaxBytes] = {i: _MaxBytes(bytes([i])) for i in range(256)}

    words: list[list[int]] = [list(w) for w in word_counts]
    counts: list[int] = [word_counts[w] for w in word_counts]

    pair_counts: dict[tuple[int, int], int] = {}
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

    for idx, word in enumerate(words):
        count = counts[idx]
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] = pair_counts.get(p, 0) + count
            pair_to_words[p].add(idx)

    # Max-heap via min-heap with negated/inverted keys.
    # Heap item: (-count, max_vocab[a], max_vocab[b], pair)
    # - -count: highest count pops first
    # - _MaxBytes wraps vocab bytes so lexicographically larger bytes sort
    #   smaller, preserving CS336 tie-break (count desc, then bytes desc)
    heap: list[tuple[int, _MaxBytes, _MaxBytes, tuple[int, int]]] = []
    for p, c in pair_counts.items():
        a, b = p
        heapq.heappush(heap, (-c, max_vocab[a], max_vocab[b], p))

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256 - len(special_tokens)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Pop the max-count pair, skipping stale entries (lazy deletion).
        # When count differs from pair_counts, re-push with the current count.
        best_pair = None
        while heap:
            neg_c, _, _, p = heap[0]
            current = pair_counts.get(p)
            if current is None:
                heapq.heappop(heap)
                continue
            if current == -neg_c:
                best_pair = p
                heapq.heappop(heap)
                break
            heapq.heappop(heap)
            a, b = p
            heapq.heappush(heap, (-current, max_vocab[a], max_vocab[b], p))

        if best_pair is None:
            break

        new_id = len(vocab)
        new_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[new_id] = new_bytes
        max_vocab[new_id] = _MaxBytes(new_bytes)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        a, b = best_pair
        affected = list(pair_to_words.get(best_pair, set()))
        changed_pairs: set[tuple[int, int]] = set()

        for idx in affected:
            word = words[idx]
            count = counts[idx]

            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p not in pair_counts:
                    continue
                pair_counts[p] -= count
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                    pair_to_words.pop(p, None)
                else:
                    pair_to_words[p].discard(idx)
                changed_pairs.add(p)

            new_word: list[int] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[idx] = new_word

            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + count
                pair_to_words[p].add(idx)
                changed_pairs.add(p)

        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)
        changed_pairs.discard(best_pair)

        # Push current count for every pair whose count changed this merge.
        # Lazy deletion skips stale entries when popping the heap.
        for p in changed_pairs:
            current = pair_counts.get(p)
            if current is not None and current > 0:
                pa, pb = p
                heapq.heappush(heap, (-current, max_vocab[pa], max_vocab[pb], p))

    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        self.pair_rank: dict[tuple[int, int], int] = {}
        self.pair_to_merged_id: dict[tuple[int, int], int] = {}
        for rank, (b1, b2) in enumerate(self.merges):
            id1 = self.inverse_vocab.get(b1)
            id2 = self.inverse_vocab.get(b2)
            merged_id = self.inverse_vocab.get(b1 + b2)
            if id1 is None or id2 is None or merged_id is None:
                continue
            self.pair_rank[(id1, id2)] = rank
            self.pair_to_merged_id[(id1, id2)] = merged_id

        self.special_token_ids: dict[str, int] = {}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes in self.inverse_vocab:
                self.special_token_ids[st] = self.inverse_vocab[st_bytes]
            else:
                new_id = len(self.vocab)
                self.vocab[new_id] = st_bytes
                self.inverse_vocab[st_bytes] = new_id
                self.special_token_ids[st] = new_id

        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = "(" + "|".join(re.escape(s) for s in sorted_specials) + ")"
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    continue
                merges.append((base64.b64decode(parts[0]), base64.b64decode(parts[1])))

        return cls(vocab, merges, special_tokens)

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        if not word_bytes:
            return []
        tokens: list[int] = [self.inverse_vocab[bytes([b])] for b in word_bytes]

        while len(tokens) >= 2:
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.pair_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break

            merged_id = self.pair_to_merged_id[(tokens[best_idx], tokens[best_idx + 1])]
            tokens = tokens[:best_idx] + [merged_id] + tokens[best_idx + 2 :]

        return tokens

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _COMPILED.finditer(text):
            ids.extend(self._encode_word(match.group().encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if self._special_pattern is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        for part in re.split(self._special_pattern, text):
            if not part:
                continue
            if part in self.special_token_ids:
                ids.append(self.special_token_ids[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode text chunks as though their concatenation were encoded once.

        The final pre-token is retained between chunks because words and
        whitespace runs can span iterable (for example, line) boundaries.
        Possible prefixes of special tokens are retained for the same reason.
        """
        buffer = ""
        for text in iterable:
            if not text:
                continue
            buffer += text

            if self._special_pattern is not None:
                while match := re.search(self._special_pattern, buffer):
                    yield from self._encode_ordinary(buffer[: match.start()])
                    yield self.special_token_ids[match.group()]
                    buffer = buffer[match.end() :]

            if not buffer:
                continue

            matches = list(_COMPILED.finditer(buffer))
            if not matches:
                continue

            retain_from = matches[-1].start()
            for special_token in self.special_tokens:
                max_prefix_length = min(len(buffer), len(special_token) - 1)
                for prefix_length in range(max_prefix_length, 0, -1):
                    if buffer.endswith(special_token[:prefix_length]):
                        partial_special_start = len(buffer) - prefix_length
                        for match in matches:
                            if match.end() > partial_special_start:
                                retain_from = min(retain_from, match.start())
                                break
                        break

            for match in matches:
                if match.end() > retain_from:
                    break
                yield from self._encode_word(match.group().encode("utf-8"))
            buffer = buffer[retain_from:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: Iterable[int]) -> str:
        chunks: list[bytes] = []
        for i in ids:
            b = self.vocab.get(i)
            if b is not None:
                chunks.append(b)
        return b"".join(chunks).decode("utf-8", errors="replace")
