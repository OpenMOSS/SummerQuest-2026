from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import BinaryIO

import regex
import tiktoken

GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


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


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[tuple[bool, str]]:
    if not special_tokens:
        return [(False, text)]

    escaped = [regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    pattern = regex.compile("|".join(escaped))

    parts: list[tuple[bool, str]] = []
    last_end = 0
    for match in pattern.finditer(text):
        if match.start() > last_end:
            parts.append((False, text[last_end : match.start()]))
        parts.append((True, match.group(0)))
        last_end = match.end()
    if last_end < len(text):
        parts.append((False, text[last_end:]))
    return parts


def _update_word_counts_from_text(
    text: str,
    *,
    special_tokens: list[str],
    pattern: regex.Pattern[str],
    word_counts: Counter[tuple[int, ...]],
) -> None:
    for is_special, part in _split_on_special_tokens(text, special_tokens):
        if is_special or not part:
            continue
        for pretoken in pattern.findall(part):
            token_bytes = pretoken.encode("utf-8")
            word = tuple(token_bytes)
            if word:
                word_counts[word] += 1


def _find_line_chunk_boundaries(file: BinaryIO, desired_num_chunks: int) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [0]

    for chunk_idx in range(1, desired_num_chunks):
        probe = chunk_idx * chunk_size
        if probe >= file_size:
            break
        file.seek(probe)
        file.readline()
        boundary = file.tell()
        if 0 < boundary < file_size:
            boundaries.append(boundary)

    boundaries.append(file_size)
    return sorted(set(boundaries))


def _count_words_in_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[int, ...]]:
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    pattern = regex.compile(GPT2_PATTERN)
    word_counts: Counter[tuple[int, ...]] = Counter()
    _update_word_counts_from_text(
        text,
        special_tokens=special_tokens,
        pattern=pattern,
        word_counts=word_counts,
    )
    return word_counts


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.id_to_token = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        special_token_to_id: dict[str, int] = {}
        special_token_bytes = {token.encode("utf-8") for token in self.special_tokens}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.token_to_id:
                special_token_to_id[token] = self.token_to_id[token_bytes]

        mergeable_ranks = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
            if token_bytes not in special_token_bytes
        }
        self.encoding = tiktoken.Encoding(
            name="cs336_bpe",
            pat_str=GPT2_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_token_to_id,
            explicit_n_vocab=len(vocab),
        )

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special=set(self.special_tokens))

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    progress = bool(kwargs.pop("progress", False))
    num_workers = max(1, int(kwargs.pop("num_workers", 1)))
    if kwargs:
        pass

    word_counts: Counter[tuple[int, ...]] = Counter()

    if num_workers > 1:
        from tqdm import tqdm

        total_bytes = os.path.getsize(input_path)
        desired_num_chunks = max(num_workers * 4, num_workers)
        with open(input_path, "rb") as f:
            boundaries = _find_line_chunk_boundaries(f, desired_num_chunks)

        chunks = [
            (boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
            if boundaries[i] < boundaries[i + 1]
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_count_words_in_chunk, input_path, start, end, special_tokens): (start, end)
                for start, end in chunks
            }
            progress_bar = tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc="pretokenize",
                disable=not progress,
            )
            with progress_bar as pbar:
                for future in as_completed(futures):
                    start, end = futures[future]
                    word_counts.update(future.result())
                    pbar.update(end - start)
    elif progress:
        pattern = regex.compile(GPT2_PATTERN)
        from tqdm import tqdm

        total_bytes = os.path.getsize(input_path)
        with open(input_path, "rb") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc="pretokenize") as pbar:
            for raw_line in f:
                pbar.update(len(raw_line))
                _update_word_counts_from_text(
                    raw_line.decode("utf-8"),
                    special_tokens=special_tokens,
                    pattern=pattern,
                    word_counts=word_counts,
                )
    else:
        pattern = regex.compile(GPT2_PATTERN)
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        _update_word_counts_from_text(
            text,
            special_tokens=special_tokens,
            pattern=pattern,
            word_counts=word_counts,
        )

    token_bytes_by_id: list[bytes] = [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    for word, count in word_counts.items():
        seen_pairs: set[tuple[int, int]] = set()
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            if pair not in seen_pairs:
                pair_to_words[pair].add(word)
                seen_pairs.add(pair)

    target_num_merges = vocab_size - len(special_tokens) - 256

    merge_iter = range(target_num_merges)
    if progress:
        merge_iter = tqdm(merge_iter, desc="bpe merges")

    for _ in merge_iter:
        if not pair_counts:
            break

        best_pair = max(
            pair_counts.items(),
            key=lambda item: (
                item[1],
                token_bytes_by_id[item[0][0]],
                token_bytes_by_id[item[0][1]],
            ),
        )[0]

        new_token_bytes = token_bytes_by_id[best_pair[0]] + token_bytes_by_id[best_pair[1]]
        new_token_id = len(token_bytes_by_id)
        token_bytes_by_id.append(new_token_bytes)
        merges.append((token_bytes_by_id[best_pair[0]], token_bytes_by_id[best_pair[1]]))

        affected_words = list(pair_to_words.pop(best_pair, set()))
        for word in affected_words:
            count = word_counts.pop(word, 0)
            if count == 0:
                continue

            for pair in zip(word, word[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

            merged_word: list[int] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    merged_word.append(new_token_id)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            new_word = tuple(merged_word)
            word_counts[new_word] += count

            seen_pairs: set[tuple[int, int]] = set()
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += count
                if pair not in seen_pairs:
                    pair_to_words[pair].add(new_word)
                    seen_pairs.add(pair)

    vocab_values: list[bytes] = [token.encode("utf-8") for token in special_tokens]
    vocab_values.extend(token_bytes_by_id)
    vocab = {token_id: token_bytes for token_id, token_bytes in enumerate(vocab_values)}
    return vocab, merges
