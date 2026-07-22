import os
import regex as re
from collections.abc import Iterable, Iterator
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import heapq

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self,
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary,
        list of merges, and (optionally) a list of special tokens.
        This function should accept the following parameters:
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        max_id = max(self.vocab.keys()) if self.vocab else -1
        
        self.special_token_to_id = {}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes in self.byte_to_id:
                self.special_token_to_id[st] = self.byte_to_id[st_bytes]
            else:
                max_id += 1
                self.vocab[max_id] = st_bytes
                self.byte_to_id[st_bytes] = max_id
                self.special_token_to_id[st] = max_id
            
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(st) for st in sorted_special_tokens]
            self.special_regex = re.compile(f"({'|'.join(escaped)})")
        else:
            self.special_regex = None
    
    def encode(self, text: str) ->list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        if self.special_regex:
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]
        ids = []
        for chunk in chunks:
            if not chunk:
                continue
                
            if chunk in self.special_token_to_id:
                ids.append(self.special_token_to_id[chunk])
            else:
                ids.extend(self._encode_chunk(chunk))
                
        return ids
    
    def _encode_chunk(self, text_chunk: str) -> list[int]:
        ids = []
        for match in self.pat.finditer(text_chunk):
            pre_token = match.group()
            b_seq = [bytes([b]) for b in pre_token.encode('utf-8')]
            
            while len(b_seq) >= 2:
                best_pair = None
                best_rank = float('inf')
                
                for i in range(len(b_seq) - 1):
                    pair = (b_seq[i], b_seq[i+1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            
                if best_pair is None:
                    break
                
                new_b_seq = []
                i = 0
                while i < len(b_seq):
                    if i < len(b_seq) - 1 and (b_seq[i], b_seq[i+1]) == best_pair:
                        new_b_seq.append(b_seq[i] + b_seq[i+1])
                        i += 2
                    else:
                        new_b_seq.append(b_seq[i])
                        i += 1
                b_seq = new_b_seq
                
            for b in b_seq:
                ids.append(self.byte_to_id[b])
        return ids    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        b_list = []
        for token_id in ids:
            if token_id in self.vocab:
                b_list.append(self.vocab[token_id])
            else:
                pass 
                
        b_text = b"".join(b_list)
        return b_text.decode("utf-8", errors="replace")

def _process_chunk(chunk_text: str) -> Counter:
    local_counter = Counter()
    for match in PAT.finditer(chunk_text):
        local_counter[match.group()] += 1
    return local_counter


def _find_chunk_boundaries(
    input_path: str | os.PathLike,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    with open(input_path, "rb") as source:
        source.seek(0, os.SEEK_END)
        file_size = source.tell()
        chunk_size = max(1, file_size // desired_num_chunks)
        boundaries = [index * chunk_size for index in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size
        for boundary_index in range(1, len(boundaries) - 1):
            source.seek(boundaries[boundary_index])
            while True:
                block_start = source.tell()
                block = source.read(4096)
                if not block:
                    boundaries[boundary_index] = file_size
                    break
                special_offset = block.find(split_special_token)
                if special_offset != -1:
                    boundaries[boundary_index] = block_start + special_offset
                    break
    return sorted(set(boundaries))


def _process_file_chunk(args: tuple[str, int, int, tuple[str, ...]]) -> Counter:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as source:
        source.seek(start)
        text = source.read(end - start).decode("utf-8")

    if not special_tokens:
        return _process_chunk(text)
    split_pattern = re.compile(f"({'|'.join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))})")
    special_token_set = set(special_tokens)
    counts = Counter()
    for piece in split_pattern.split(text):
        if piece and piece not in special_token_set:
            counts.update(_process_chunk(piece))
    return counts


class _DescendingPair:
    """Reverse the byte-pair ordering so heapq implements the required max tie-break."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_DescendingPair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DescendingPair) and self.pair == other.pair


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter:
    input_path = os.fspath(input_path)
    worker_count = min(cpu_count(), 16)
    if special_tokens and worker_count > 1:
        split_token = max(special_tokens, key=len).encode("utf-8")
        boundaries = _find_chunk_boundaries(input_path, worker_count, split_token)
        tasks = [
            (input_path, start, end, tuple(special_tokens))
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        counts = Counter()
        with Pool(processes=len(tasks)) as pool:
            for chunk_counts in pool.imap_unordered(_process_file_chunk, tasks):
                counts.update(chunk_counts)
        return counts

    with open(input_path, encoding="utf-8") as source:
        text = source.read()
    if not special_tokens:
        return _process_chunk(text)
    return _process_file_chunk((input_path, 0, os.path.getsize(input_path), tuple(special_tokens)))
 
 
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    byte_to_id = {b: i for i, b in vocab.items()}
 
    max_id = 255
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in byte_to_id:
            max_id += 1
            vocab[max_id] = st_bytes
            byte_to_id[st_bytes] = max_id
 
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []
 
    pre_token_counts = _count_pretokens(input_path, special_tokens)

    words: list[tuple[bytes, ...]] = []
    word_counts: list[int] = []
    for word, count in pre_token_counts.items():
        b_seq = tuple(bytes([b]) for b in word.encode("utf-8"))
        words.append(b_seq)
        word_counts.append(count)

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, (word_seq, count) in enumerate(zip(words, word_counts)):
        word_pair_counts = Counter(zip(word_seq, word_seq[1:]))
        for pair, occurrences in word_pair_counts.items():
            pair_counts[pair] += occurrences * count
            pair_to_words[pair].add(word_id)

    pair_heap: list[tuple[int, _DescendingPair]] = [
        (-count, _DescendingPair(pair)) for pair, count in pair_counts.items() if count > 0
    ]
    heapq.heapify(pair_heap)
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        while pair_heap:
            negative_count, descending_pair = heapq.heappop(pair_heap)
            candidate_pair = descending_pair.pair
            if pair_counts.get(candidate_pair, 0) == -negative_count and negative_count < 0:
                break
        else:
            break

        best_pair = candidate_pair
        merged_token = best_pair[0] + best_pair[1]
        max_id += 1
        vocab[max_id] = merged_token
        merges.append(best_pair)

        affected_word_ids = list(pair_to_words.get(best_pair, ()))
        changed_pairs: set[tuple[bytes, bytes]] = set()
        for word_id in affected_word_ids:
            word_seq = words[word_id]
            count = word_counts[word_id]
            old_pair_counts = Counter(zip(word_seq, word_seq[1:]))
            for pair, occurrences in old_pair_counts.items():
                pair_counts[pair] -= occurrences * count
                pair_to_words[pair].discard(word_id)
                changed_pairs.add(pair)

            new_seq = []
            i = 0
            while i < len(word_seq):
                if i < len(word_seq) - 1 and (word_seq[i], word_seq[i + 1]) == best_pair:
                    new_seq.append(merged_token)
                    i += 2
                else:
                    new_seq.append(word_seq[i])
                    i += 1
            new_word_seq = tuple(new_seq)
            words[word_id] = new_word_seq

            new_pair_counts = Counter(zip(new_word_seq, new_word_seq[1:]))
            for pair, occurrences in new_pair_counts.items():
                pair_counts[pair] += occurrences * count
                pair_to_words[pair].add(word_id)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            count = pair_counts[pair]
            if count > 0:
                heapq.heappush(pair_heap, (-count, _DescendingPair(pair)))

    return vocab, merges
