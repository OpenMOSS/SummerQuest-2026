import regex as re
from collections import Counter
import heapq
from collections import defaultdict
from typing import BinaryIO
import os
from multiprocessing import Pool


def process_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]
    del text      
    local_counts = Counter()
    for chunk in chunks:
        for pre_token in re.findall(PAT, chunk):
            pro_token = pre_token.encode("utf-8")
            pro_token_split = [bytes([b]) for b in pro_token]
            local_counts[tuple(pro_token_split)] += 1    
    return local_counts    

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def train_bpe(input_path, vocab_size, special_tokens):
    file_size = os.path.getsize(input_path)
    num_processes = 1 if file_size < 1_000_000 else os.cpu_count()
    if num_processes is None:
        num_processes = os.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    args = [(input_path, s, e, special_tokens) 
            for s, e in zip(boundaries[:-1], boundaries[1:])]
    
    with Pool(num_processes) as pool:
        results = pool.map(process_chunk, args)
    
    counts = Counter()
    for c in results:
        counts.update(c)  
    
    pair_counts = Counter()
    pair_to_words = defaultdict(set)     
    for word, freq in counts.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] += freq
            pair_to_words[p].add(word)   

    vocab = {}
    merges = []
    for t in range(256):
        vocab[t] = bytes([t])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    
    heap = [(-count, RevKey(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    
    while len(vocab) < vocab_size:   
        best_pair = None
        while heap:
            neg_count, _, pair = heapq.heappop(heap)
            if pair_counts.get(pair) == -neg_count:   
                best_pair = pair
                break
        if best_pair is None:
            break
   
        affected = list(pair_to_words.pop(best_pair, set()))

        for word in affected:
            freq = counts[word] 
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                else:
                    heapq.heappush(heap, (-pair_counts[p], RevKey(p), p))
                pair_to_words[p].discard(word)
            del counts[word]
            
            new_word = merge_pairs(word, best_pair)
            
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += freq
                heapq.heappush(heap, (-pair_counts[p], RevKey(p), p))
                pair_to_words[p].add(new_word)
            counts[new_word] += freq
        
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        
    return vocab, merges

def merge_pairs(word, pair):
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

class RevKey:
    __slots__ = ("pair",)
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair    
    
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


    
    
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}   
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.cache = {}
        
    def _merge(self, word):
        while len(word) >= 2:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            if best not in self.ranks:
                break
            word = merge_pairs(word, best)
        return word
    
    def _split(self, text):
        if not self.special_tokens:
            return [text]
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(s) for s in sorted_specials)
  
        return re.split(f"({pattern})", text)
    
    def encode(self, text: str) -> list[int]:
        ids = []
        splits = self._split(text)
        for split in splits:
            if split in self.special_tokens:
                ids.append(self.byte_to_id[split.encode("utf-8")])
                continue
            for pre_token in re.findall(PAT, split):
                ids.extend(self._encode_word(pre_token))
        return ids

    def decode(self, ids: list[int]) -> str:
        bytes_seq = b"".join([self.vocab[i] for i in ids])    
        return bytes_seq.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        for line in iterable:
            yield from self.encode(line)
    
    def _encode_word(self, pre_token: str) -> list[int]:
        if pre_token in self.cache:
            return self.cache[pre_token]
        word = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
        word = self._merge(word)
        result = [self.byte_to_id[tok] for tok in word]
        self.cache[pre_token] = result
        return result