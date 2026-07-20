from typing import Iterable, Iterator, List, Tuple, Dict
import heapq
import json
import multiprocessing as mp
import regex as re


GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """Split text on special tokens, removing them."""
    if not special_tokens:
        return [text]
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
    return re.split(special_pattern, text)


def _preprocess_corpus(
    text: str,
    special_tokens: List[str],
    pretokenizer_pattern: str,
) -> Iterator[List[int]]:
    """
    Split the corpus into pre-tokens then yield each pre-token as a list of byte ids.

    Steps:
        1. Split text on special_tokens (remove them from the train data)
        2. Apply GPT-2 pre-tokenization regex to each chunk
        3. Encode each pre-token as UTF-8 bytes and return byte ids.

    Args:
        text: raw training corpus.
        special_tokens: special tokens to exclude from the BPE learning
        pretokenizer_pattern: GPT-2 regex string

    Yields: List of byte ids, one per pre-token.
    """
    chunks = _split_on_special_tokens(text, special_tokens)

    pretoken_regex = re.compile(pretokenizer_pattern)
    for chunk in chunks:
        for match in pretoken_regex.finditer(chunk):
            pt = match.group(0)
            if pt:
                yield list(pt.encode("utf-8"))


def _preprocess_chunks_to_counter(
    args: Tuple[List[str], List[str], str],
) -> Dict[Tuple[int, ...], int]:
    """Worker: pre-tokenize chunks and return a frequency counter of unique pre-tokens."""
    chunks, special_tokens, pretokenizer_pattern = args
    counter: Dict[Tuple[int, ...], int] = {}
    for chunk in chunks:
        for seq in _preprocess_corpus(chunk, special_tokens, pretokenizer_pattern):
            key = tuple(seq)
            counter[key] = counter.get(key, 0) + 1
    return counter


def _iter_document_chunks(
    path: str,
    special_tokens: List[str],
    chunk_size: int = 100_000_000,
) -> Iterator[str]:
    """Stream document-sized chunks from a file, splitting on special tokens.

    If no special tokens are provided, the entire file is yielded as one chunk.
    """
    if not special_tokens:
        with open(path, encoding="utf-8") as f:
            yield f.read()
        return

    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)

    buffer = ""
    with open(path, encoding="utf-8") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            buffer += data
            parts = re.split(special_pattern, buffer)
            # All parts except the last one end on a special-token boundary.
            for part in parts[:-1]:
                if part:
                    yield part
            buffer = parts[-1]
        if buffer:
            yield buffer


def _build_unique_pretokens_from_path(
    path: str,
    special_tokens: List[str],
    pretokenizer_pattern: str,
    num_workers: int,
    chunk_size: int = 100_000_000,
) -> Dict[Tuple[int, ...], int]:
    """Return a frequency counter of unique pre-token byte-id sequences.

    Processes the file in streaming chunks so the whole corpus need not fit in
    memory as a single Python string.
    """
    counter: Dict[Tuple[int, ...], int] = {}

    # Stream document chunks and process in worker-sized batches.
    batch: List[str] = []
    batch_count = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        for doc_chunk in _iter_document_chunks(path, special_tokens, chunk_size):
            batch.append(doc_chunk)
            if len(batch) >= num_workers * 4:
                # Split batch across workers round-robin.
                groups: List[List[str]] = [[] for _ in range(num_workers)]
                for i, chunk in enumerate(batch):
                    groups[i % num_workers].append(chunk)
                args_list = [
                    (group, special_tokens, pretokenizer_pattern)
                    for group in groups
                    if group
                ]
                for partial in pool.imap_unordered(_preprocess_chunks_to_counter, args_list):
                    for key, count in partial.items():
                        counter[key] = counter.get(key, 0) + count
                batch = []
                batch_count += 1

        if batch:
            groups = [[] for _ in range(num_workers)]
            for i, chunk in enumerate(batch):
                groups[i % num_workers].append(chunk)
            args_list = [
                (group, special_tokens, pretokenizer_pattern)
                for group in groups
                if group
            ]
            for partial in pool.imap_unordered(_preprocess_chunks_to_counter, args_list):
                for key, count in partial.items():
                    counter[key] = counter.get(key, 0) + count

    return counter


def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_workers: int | None = None,
    min_frequency: int = 1,
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on input_path.

    Returns:
        vocab: dict mapping token_id -> bytes
        merges: list of (token_a, token_b) pairs, in order of merging
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    # Initial vocabulary: 256 bytes + special tokens.
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    merges: List[Tuple[bytes, bytes]] = []

    # Build unique pre-token frequency map (streaming to bound memory).
    unique_counts = _build_unique_pretokens_from_path(
        input_path, special_tokens, GPT2_PRETOKENIZER_PATTERN, num_workers
    )

    if min_frequency > 1:
        unique_counts = {k: v for k, v in unique_counts.items() if v >= min_frequency}

    if not unique_counts:
        return vocab, merges

    # Single-pass: build linked lists + init pair stats from unique pre-tokens.
    unique_map: Dict[Tuple[int, ...], int] = {}
    words_token: List[List[int]] = []
    words_prev: List[List[int]] = []
    words_next: List[List[int]] = []
    words_deleted: List[List[bool]] = []
    words_head: List[int] = []
    weights: List[int] = []
    words_pairs: List[List[Tuple[int, int]]] = []

    pair_counts: Dict[Tuple[int, int], int] = {}
    pair_positions: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def add_pair(word_idx: int, left_idx: int, pair: Tuple[int, int]) -> None:
        pair_counts[pair] = pair_counts.get(pair, 0) + weights[word_idx]
        pair_positions.setdefault(pair, []).append((word_idx, left_idx))

    for seq_tuple, weight in unique_counts.items():
        word_idx = len(words_token)
        unique_map[seq_tuple] = word_idx
        seq = list(seq_tuple)
        n = len(seq)
        words_token.append(seq)
        words_prev.append(list(range(-1, n - 1)))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        words_next.append(nxt)
        words_deleted.append([False] * n)
        words_head.append(0)
        weights.append(weight)

        pairs: List[Tuple[int, int]] = []
        left_idx = words_head[word_idx]
        while left_idx != -1:
            right_idx = words_next[word_idx][left_idx]
            if right_idx == -1:
                break
            pair = (words_token[word_idx][left_idx], words_token[word_idx][right_idx])
            pairs.append(pair)
            add_pair(word_idx, left_idx, pair)
            left_idx = right_idx
        words_pairs.append(pairs)

    # Heap for fast max-pair selection.
    class _HeapEntry:
        __slots__ = ("count", "pair", "byte_pair")

        def __init__(self, count: int, pair: Tuple[int, int]):
            self.count = count
            self.pair = pair
            self.byte_pair = (vocab[pair[0]], vocab[pair[1]])

        def __lt__(self, other: "_HeapEntry") -> bool:
            if self.count != other.count:
                return self.count > other.count  # larger count first
            return self.byte_pair > other.byte_pair  # lexicographically larger byte pair first

    heap: List[_HeapEntry] = []

    def push_pair(pair: Tuple[int, int]) -> None:
        count = pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(heap, _HeapEntry(count, pair))

    for pair in pair_counts:
        push_pair(pair)

    # BPE training loop.
    while len(vocab) < vocab_size:
        # Pop stale entries.
        while heap:
            entry = heap[0]
            current_count = pair_counts.get(entry.pair, 0)
            if current_count == entry.count and current_count > 0:
                break
            heapq.heappop(heap)
        else:
            break

        entry = heapq.heappop(heap)
        best_pair = entry.pair
        count = entry.count
        if count <= 0:
            break

        a, b = best_pair
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        positions = pair_positions.get(best_pair, [])
        pair_positions[best_pair] = []
        pair_counts[best_pair] = 0

        changed_pairs: List[Tuple[int, int]] = []

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

            # Remove old adjacent pairs.
            if prev_idx != -1:
                old_pair = (prev_token, a)
                pair_counts[old_pair] = pair_counts.get(old_pair, 0) - weights[word_idx]
                changed_pairs.append(old_pair)
            if next_idx != -1:
                old_pair = (b, next_token)
                pair_counts[old_pair] = pair_counts.get(old_pair, 0) - weights[word_idx]
                changed_pairs.append(old_pair)

            # Mark old nodes as deleted and create new merged node.
            words_deleted[word_idx][left_idx] = True
            words_deleted[word_idx][right_idx] = True

            new_idx = len(words_token[word_idx])
            words_token[word_idx].append(new_token_id)
            words_prev[word_idx].append(prev_idx)
            words_next[word_idx].append(next_idx)
            words_deleted[word_idx].append(False)

            # Re-wire neighbors.
            if prev_idx != -1:
                words_next[word_idx][prev_idx] = new_idx
            else:
                words_head[word_idx] = new_idx
            if next_idx != -1:
                words_prev[word_idx][next_idx] = new_idx

            # Add new adjacent pairs.
            if prev_idx != -1:
                new_pair = (prev_token, new_token_id)
                add_pair(word_idx, prev_idx, new_pair)
                changed_pairs.append(new_pair)
            if next_idx != -1:
                new_pair = (new_token_id, next_token)
                add_pair(word_idx, new_idx, new_pair)
                changed_pairs.append(new_pair)

        for pair in changed_pairs:
            push_pair(pair)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Build reverse vocab for special tokens.
        self.special_token_ids: Dict[str, int] = {}
        for token_id, token_bytes in self.vocab.items():
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if token_str in self.special_tokens:
                self.special_token_ids[token_str] = token_id

        # Build reverse mapping from token bytes to id.
        token_to_id = {bytes_val: token_id for token_id, bytes_val in self.vocab.items()}
        self.token_to_id: Dict[bytes, int] = token_to_id

        # Build merge rank and pair-to-merged-id mapping for fast encoding.
        self.merge_rank: Dict[Tuple[int, int], int] = {}
        self.merge_pair_to_id: Dict[Tuple[int, int], int] = {}
        for rank, (a, b) in enumerate(self.merges):
            aid = token_to_id[a]
            bid = token_to_id[b]
            self.merge_rank[(aid, bid)] = rank
            merged_bytes = a + b
            merged_id = token_to_id[merged_bytes]
            self.merge_pair_to_id[(aid, bid)] = merged_id

        # Pre-compile regexes for encoding.
        self._pretoken_regex = re.compile(GPT2_PRETOKENIZER_PATTERN)
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        self._special_split_regex = re.compile(f"({special_pattern})") if special_tokens else None

        # Cache for frequently-occurring pre-token encodings.
        self._pretoken_cache: Dict[str, Tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_path, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = json.loads(line)
                merges.append((bytes(a), bytes(b)))

        return cls(vocab, merges, special_tokens=special_tokens)

    def save(self, vocab_path: str, merges_path: str) -> None:
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({str(k): list(v) for k, v in self.vocab.items()}, f)
        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(json.dumps([list(a), list(b)]) + "\n")

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode a text chunk that contains no special tokens."""
        pre_tokens = self._pretoken_regex.findall(text)

        result: List[int] = []
        for pt in pre_tokens:
            if not pt:
                continue
            cached = self._pretoken_cache.get(pt)
            if cached is not None:
                result.extend(cached)
                continue

            seq = [self.token_to_id[bytes([b])] for b in pt.encode("utf-8")]

            # Apply merges greedily by earliest merge rank, modifying in-place.
            while len(seq) > 1:
                best_rank = float("inf")
                best_pos = -1
                for i in range(len(seq) - 1):
                    rank = self.merge_rank.get((seq[i], seq[i + 1]))
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pos = i
                if best_pos == -1:
                    break
                merged_id = self.merge_pair_to_id[(seq[best_pos], seq[best_pos + 1])]
                seq[best_pos] = merged_id
                del seq[best_pos + 1]

            encoded = tuple(seq)
            self._pretoken_cache[pt] = encoded
            result.extend(encoded)
        return result

    def encode(self, text: str) -> List[int]:
        if not self.special_tokens:
            return self._encode_chunk(text)

        parts = self._special_split_regex.split(text)
        result: List[int] = []
        for part in parts:
            if part in self.special_token_ids:
                result.append(self.special_token_ids[part])
            else:
                result.extend(self._encode_chunk(part))
        return result

    def decode(self, token_ids: List[int]) -> str:
        all_bytes = b"".join(self.vocab[i] for i in token_ids)
        return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
