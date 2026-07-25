import regex
import os
import collections
import multiprocessing
from dataclasses import dataclass
import heapq

from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

PARALLEL_PRETOKENIZE_MIN_BYTES=1_000_000
MAX_PRETOKENIZE_PROCESSES=8
PRETOKENIZE_TASK_TARGET_BYTES=128 * 1024**2

Pair = tuple[bytes,bytes]
Position = tuple[int,int,int]
WordTokens = list[bytes | None]
PreTokenCounts = collections.Counter[bytes]

@dataclass(frozen=True,slots=True)
class PairHeapEntry:
    pair: Pair
    count: int

    def __lt__(self,other:"PairHeapEntry") -> bool:
        return (self.count,self.pair) >(
            other.count,
            other.pair,
        )

def build_pair_heap(
    pair_counts: collections.Counter[Pair],
) -> list[PairHeapEntry]:
    pair_heap =[
        PairHeapEntry(pair=pair, count=count)
        for pair,count in pair_counts.items()
        if count >0
    ]

    heapq.heapify(pair_heap)
    return pair_heap

def pop_best_pair(
    pair_heap: list[PairHeapEntry],
    pair_counts: collections.Counter[Pair],
) -> Pair | None:
    while pair_heap:
        entry = heapq.heappop(pair_heap)
        current_count=pair_counts.get(entry.pair)

        if current_count is None:
            continue

        if current_count != entry.count:
            continue

        return entry.pair

    return None

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks:int,
        split_special_token:bytes,
)-> list[int]:
    assert isinstance(split_special_token,bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries=[
        index*chunk_size
        for index in range(desired_num_chunks +1)
    ]
    chunk_boundaries[-1]=file_size

    mini_chunk_size = 4096

    for boundary_index in range(1, len(chunk_boundaries)-1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk ==b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                chunk_boundaries[boundary_index] = (
                    initial_position + found_at
                )
                break

            initial_position+=mini_chunk_size

    return sorted(set(chunk_boundaries))

def pre_tokenize(
        text:str,
        special_tokens: list[str]
 ) -> PreTokenCounts:
    pre_token_counts: PreTokenCounts = collections.Counter()

    if special_tokens:
        special_pattern = "|".join(
            regex.escape(token) for token in special_tokens
        )
        text_segments = regex.split(special_pattern, text)
    else:
        text_segments = [text]

    for segment in text_segments:
        for match in regex.finditer(PAT, segment):
            pre_token=match.group(0).encode("utf-8")
            pre_token_counts[pre_token] +=1

    return pre_token_counts

def pre_tokenize_chunk(
        input_path:str | os.PathLike,
        start: int,
        end: int,
        special_tokens: list[str],
) -> PreTokenCounts:
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        chunk_bytes=input_file.read(end-start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    del chunk_bytes

    return pre_tokenize(chunk_text, special_tokens)

def pre_tokenize_chunk_task(
        task: tuple[
            str | os.PathLike,
            int,
            int,
            list[str],
        ],
) -> PreTokenCounts:
    return pre_tokenize_chunk(*task)

def parallel_pre_tokenize(
        input_path: str | os.PathLike,
        special_tokens: list[str],
        num_processes: int,
        desired_num_chunks: int | None = None,
) -> PreTokenCounts:
    if num_processes <1 :
        raise ValueError("num_processes 必须至少为1")
    if desired_num_chunks is None:
        desired_num_chunks = num_processes
    if desired_num_chunks < 1:
        raise ValueError("desired_num_chunks 必须至少为1")
    if not special_tokens:
        with open(input_path, "r", encoding="utf-8") as input_file:
            text=input_file.read()
        return pre_tokenize(text,special_tokens)

    split_special_token=special_tokens[0].encode("utf-8")

    with open(input_path, "rb") as input_file:
        boundaries = find_chunk_boundaries(
            input_file,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_special_token,
        )

    tasks =[
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1],boundaries[1:])
    ]

    if not tasks:
        return collections.Counter()

    worker_count = min(num_processes, len(tasks))

    pre_token_counts=collections.Counter()

    with multiprocessing.Pool(processes=worker_count) as pool:
        chunk_counts = pool.imap_unordered(
            pre_tokenize_chunk_task,
            tasks,
            chunksize=1,
        )

        for chunk_count in chunk_counts:
            pre_token_counts.update(chunk_count)

    return pre_token_counts

def build_merge_index(
        pre_token_counts: PreTokenCounts,
) -> tuple[
    list[WordTokens],
    list[int],
    collections.Counter[Pair],
    dict[Pair,set[Position]],
]:
    words: list[WordTokens] = []
    word_frequencies:list[int] = []
    pair_counts: collections.Counter[Pair] = collections.Counter()
    pair_positions: dict[Pair,set[Position]] = {}

    word_id = 0

    while pre_token_counts:
        pre_token, frequency = pre_token_counts.popitem()

        word = [
            bytes([byte_value])
            for byte_value in pre_token
        ]
        words.append(word)
        word_frequencies.append(frequency)

        for left_index in range(len(word)-1):
            right_index = left_index+1

            pair: Pair=(
                word[left_index],
                word[right_index],
            )

            position: Position=(
                word_id,
                left_index,
                right_index,
            )

            pair_counts[pair]+=frequency
            pair_positions.setdefault(pair,set()).add(position)

        word_id += 1

    pre_token_counts.clear()

    return (
        words,
        word_frequencies,
        pair_counts,
        pair_positions,
    )

def get_next_index(
        word: WordTokens,
        index: int,
) -> int | None:
    for candidate_index in range(index +1,len(word)):
        if word[candidate_index] is not None:
            return candidate_index

    return None

def get_previous_index(
        word:WordTokens,
        index: int,
) -> int | None:
    for candidate_index in range(index -1, -1, -1):
        if word[candidate_index] is not None:
            return candidate_index
    return None

def remove_pair_position(
        pair:Pair,
        position: Position,
        frequency: int,
        pair_counts: collections.Counter[Pair],
        pair_positions: dict[Pair,set[Position]],
) -> None:
    positions = pair_positions[pair]
    positions.remove(position)
    pair_counts[pair]-=frequency

    if not positions:
        assert pair_counts[pair]==0
        del pair_positions[pair]
        del pair_counts[pair]

def add_pair_position(
        pair: Pair,
        position: Position,
        frequency: int,
        pair_counts: collections.Counter[Pair],
        pair_positions: dict[Pair,set[Position]],
) -> None:
    positions = pair_positions.setdefault(pair,set())
    assert position not in positions

    positions.add(position)
    pair_counts[pair] += frequency

def merge_indexed_pair(
    best_pair: Pair,
    words: list[WordTokens],
    word_frequencies: list[int],
    pair_counts: collections.Counter[Pair],
    pair_positions: dict[Pair,set[Position]],
) -> set[Pair]:
    changed_pairs: set[Pair] = set()
    merged_token = best_pair[0]+best_pair[1]
    positions=sorted(pair_positions[best_pair])

    for word_id, left_index, right_index in positions:
        word = words[word_id]

        if(
            word[left_index] != best_pair[0]
            or word[right_index] !=best_pair[1]
            or get_next_index(word, left_index) != right_index
        ):
            continue

        frequency = word_frequencies[word_id]

        previous_index = get_previous_index(word,left_index)
        next_index = get_next_index(word,right_index)

        if previous_index is not None:
            previous_token =word[previous_index]
            assert previous_token is not None

            previous_pair: Pair =(
                previous_token,
                best_pair[0],
            )

            previous_position: Position=(
                word_id,
                previous_index,
                left_index
            )

            changed_pairs.add(previous_pair)
            remove_pair_position(
                previous_pair,
                previous_position,
                frequency,
                pair_counts,
                pair_positions,
            )

        current_position: Position =(
            word_id,
            left_index,
            right_index,
        )

        changed_pairs.add(best_pair)
        remove_pair_position(
            best_pair,
            current_position,
            frequency,
            pair_counts,
            pair_positions,
        )

        if next_index is not None:
            next_token =word[next_index]
            assert next_token is not None

            next_pair: Pair =(
                best_pair[1],
                next_token,

            )

            next_position: Position=(
                word_id,
                right_index,
                next_index,
            )


            changed_pairs.add(next_pair)
            remove_pair_position(
                next_pair,
                next_position,
                frequency,
                pair_counts,
                pair_positions,
            )

        word[left_index] = merged_token
        word[right_index]=None

        if previous_index is not None:
            previous_token=word[previous_index]
            assert previous_token is not None

            new_previous_pair: Pair =(
                previous_token,
                merged_token,
            )

            new_previous_position: Position =(
                word_id,
                previous_index,
                left_index,
            )

            changed_pairs.add(new_previous_pair)
            add_pair_position(
                new_previous_pair,
                new_previous_position,
                frequency,
                pair_counts,
                pair_positions,
            )

        if next_index is not None:
            next_token = word[next_index]
            assert next_token is not None

            new_next_pair: Pair = (
                merged_token,
                next_token,
            )
            new_next_position: Position = (
                word_id,
                left_index,
                next_index,
            )

            changed_pairs.add(new_next_pair)
            add_pair_position(
                new_next_pair,
                new_next_position,
                frequency,
                pair_counts,
                pair_positions,
            )
    return changed_pairs

def count_pairs(
        pre_token_counts: collections.Counter[tuple[bytes,...]]
) -> collections.Counter[tuple[bytes,bytes]]:
        pair_counts: collections.Counter[tuple[bytes,bytes]] = collections.Counter()
        for token_tuple, frequency in pre_token_counts.items():
            for index in range(len(token_tuple)-1):
                pair = (
                    token_tuple[index],
                    token_tuple[index+1],
                )
                pair_counts[pair] += frequency
        return pair_counts

def find_best_pair(
        pair_counts: collections.Counter[tuple[bytes,bytes]]
) -> tuple[bytes,bytes]:
    best_pair=max(
        pair_counts,
        key=lambda pair: (pair_counts[pair],pair),
    )
    return best_pair

def merge_pair(
        pre_token_counts:collections.Counter[tuple[bytes,...]],
        best_pair:tuple[bytes,bytes],
) -> collections.Counter[tuple[bytes,...]]:
    merged_counts: collections.Counter[tuple[bytes,...]]=collections.Counter()
    merged_token = best_pair[0]+best_pair[1]
    for token_tuple, frequency in pre_token_counts.items():
        contains_best_pair = any(
            token_tuple[index]==best_pair[0]
            and token_tuple[index+1] == best_pair[1]
            for index in range(len(token_tuple)-1)
        )
        if not contains_best_pair:
            merged_counts[token_tuple]+=frequency
            continue
        merged_tuple:list[bytes]=[]
        index = 0
        while index < len(token_tuple):
            if index < len(token_tuple)-1 and token_tuple[index:index+2]==best_pair:
                merged_tuple.append(merged_token)
                index+=2
            else:
                merged_tuple.append(token_tuple[index])
                index+=1
        merged_counts[tuple(merged_tuple)]+=frequency
    return merged_counts

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:

    vocab: dict[int,bytes]={}
    for byte_value in range(256):
        vocab[byte_value]=bytes([byte_value])

    for special_token in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = special_token.encode("utf-8")

    merges: list[tuple[bytes,bytes]]=[]

    input_size=os.path.getsize(input_path)

    if input_size >=PARALLEL_PRETOKENIZE_MIN_BYTES:
        available_cpus = os.cpu_count() or 1
        num_processes = min(
            MAX_PRETOKENIZE_PROCESSES,
            available_cpus,
        )
        desired_num_chunks = max(
            num_processes,
            (
                input_size
                + PRETOKENIZE_TASK_TARGET_BYTES
                - 1
            ) // PRETOKENIZE_TASK_TARGET_BYTES,
        )
        pre_token_counts=parallel_pre_tokenize(
            input_path=input_path,
            special_tokens=special_tokens,
            num_processes=num_processes,
            desired_num_chunks=desired_num_chunks,
        )
    else:
        with open(input_path,"r", encoding="utf-8") as input_file:
            text = input_file.read()

        pre_token_counts=pre_tokenize(text,special_tokens)


    (
        words,
        word_frequencies,
        pair_counts,
        pair_positions,
    ) = build_merge_index(pre_token_counts)

    pair_heap = build_pair_heap(pair_counts)

    while len(vocab)<vocab_size:
        best_pair=pop_best_pair(
            pair_heap,
            pair_counts,
        )
        if best_pair is None:
            break
        merges.append(best_pair)

        merged_token=best_pair[0] + best_pair[1]
        vocab[len(vocab)]=merged_token


        changed_pairs= merge_indexed_pair(
            best_pair,
            words,
            word_frequencies,
            pair_counts,
            pair_positions,
        )

        for pair in changed_pairs:
            current_count=pair_counts.get(pair)

            if current_count is not None and current_count>0:
                heapq.heappush(
                    pair_heap,
                    PairHeapEntry(
                        pair=pair,
                        count=current_count,
                    ),
                )

    return vocab,merges
