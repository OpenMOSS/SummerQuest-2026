from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import MutableMapping

import regex as re


GPT2_PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PARALLEL_PRETOKENIZATION_THRESHOLD_BYTES = 64 * 1024 * 1024
MAX_AUTO_PRETOKENIZATION_PROCESSES = 8


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    special_pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return re.split(special_pattern, text)


def _merge_token_sequence(
    tokens: tuple[int, ...],
    pair: tuple[int, int],
    merged_token_id: int,
) -> tuple[int, ...]:
    merged: list[int] = []
    i = 0
    first, second = pair
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == first and tokens[i + 1] == second:
            merged.append(merged_token_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return tuple(merged)


def _count_pretokens_in_text(
    text: str,
    special_tokens: list[str],
    byte_token_ids: tuple[int, ...],
) -> dict[tuple[int, ...], int]:
    pattern = re.compile(GPT2_PRETOKENIZATION_PATTERN)

    counts: dict[tuple[int, ...], int] = {}
    for segment in _split_on_special_tokens(text, special_tokens):
        for match in pattern.finditer(segment):
            token_bytes = match.group(0).encode("utf-8")
            token = tuple(byte_token_ids[byte] for byte in token_bytes)
            counts[token] = counts.get(token, 0) + 1
    return counts


def _add_counts(
    destination: MutableMapping[tuple[int, ...], int],
    source: MutableMapping[tuple[int, ...], int],
) -> None:
    for token, count in source.items():
        destination[token] = destination.get(token, 0) + count


def _find_chunk_boundaries(
    input_path: str | os.PathLike,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        if desired_num_chunks <= 1 or file_size == 0 or not split_special_tokens:
            return [0, file_size]

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096
        max_special_token_length = max(len(token) for token in split_special_tokens)
        overlap_size = max(0, max_special_token_length - 1)

        for boundary_index in range(1, len(boundaries) - 1):
            initial_position = boundaries[boundary_index]
            f.seek(initial_position)
            current_position = initial_position
            overlap = b""

            while True:
                chunk = f.read(mini_chunk_size)
                if chunk == b"":
                    boundaries[boundary_index] = file_size
                    break

                search_buffer = overlap + chunk
                search_buffer_start = current_position - len(overlap)
                found_positions = [
                    search_buffer.find(special_token)
                    for special_token in split_special_tokens
                    if search_buffer.find(special_token) != -1
                ]

                if found_positions:
                    boundaries[boundary_index] = search_buffer_start + min(found_positions)
                    break

                overlap = search_buffer[-overlap_size:] if overlap_size else b""
                current_position += len(chunk)

    return sorted(set(boundaries))


def _count_pretokens_in_chunk(
    args: tuple[str | os.PathLike, int, int, list[str], tuple[int, ...]],
) -> dict[tuple[int, ...], int]:
    input_path, start, end, special_tokens, byte_token_ids = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens_in_text(text, special_tokens, byte_token_ids)


def _resolve_num_processes(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None,
) -> int:
    if not special_tokens:
        return 1

    if num_processes is not None:
        return max(1, num_processes)

    file_size = os.path.getsize(input_path)
    if file_size < PARALLEL_PRETOKENIZATION_THRESHOLD_BYTES:
        return 1

    return max(1, min(os.cpu_count() or 1, MAX_AUTO_PRETOKENIZATION_PROCESSES))


def _pretoken_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    byte_token_ids: tuple[int, ...],
    num_processes: int | None,
) -> dict[tuple[int, ...], int]:
    resolved_num_processes = _resolve_num_processes(input_path, special_tokens, num_processes)

    if resolved_num_processes == 1:
        with open(input_path, encoding="utf-8") as f:
            return _count_pretokens_in_text(f.read(), special_tokens, byte_token_ids)

    split_special_tokens = [token.encode("utf-8") for token in special_tokens]
    boundaries = _find_chunk_boundaries(input_path, resolved_num_processes, split_special_tokens)
    chunk_args = [
        (input_path, start, end, special_tokens, byte_token_ids)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    if len(chunk_args) <= 1:
        return _count_pretokens_in_chunk(chunk_args[0]) if chunk_args else {}

    merged_counts: dict[tuple[int, ...], int] = {}
    with mp.Pool(processes=min(resolved_num_processes, len(chunk_args))) as pool:
        for counts in pool.imap_unordered(_count_pretokens_in_chunk, chunk_args):
            _add_counts(merged_counts, counts)
    return merged_counts


def _add_pair_counts(
    tokens: tuple[int, ...],
    count: int,
    pair_counts: MutableMapping[tuple[int, int], int],
    pair_to_tokens: MutableMapping[tuple[int, int], set[tuple[int, ...]]],
) -> None:
    if len(tokens) < 2:
        return

    seen: set[tuple[int, int]] = set()
    previous = tokens[0]
    for current in tokens[1:]:
        pair = (previous, current)
        pair_counts[pair] = pair_counts.get(pair, 0) + count
        if pair not in seen:
            pair_to_tokens.setdefault(pair, set()).add(tokens)
            seen.add(pair)
        previous = current


def _remove_pair_counts(
    tokens: tuple[int, ...],
    count: int,
    pair_counts: MutableMapping[tuple[int, int], int],
    pair_to_tokens: MutableMapping[tuple[int, int], set[tuple[int, ...]]],
) -> None:
    if len(tokens) < 2:
        return

    seen: set[tuple[int, int]] = set()
    previous = tokens[0]
    for current in tokens[1:]:
        pair = (previous, current)
        new_count = pair_counts[pair] - count
        if new_count:
            pair_counts[pair] = new_count
        else:
            del pair_counts[pair]

        if pair not in seen:
            tokens_with_pair = pair_to_tokens[pair]
            tokens_with_pair.discard(tokens)
            if not tokens_with_pair:
                del pair_to_tokens[pair]
            seen.add(pair)
        previous = current


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    special_tokens = [] if special_tokens is None else list(dict.fromkeys(special_tokens))

    vocab: dict[int, bytes] = {}
    token_to_id: dict[bytes, int] = {}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in token_to_id:
            token_to_id[token_bytes] = len(vocab)
            vocab[len(vocab)] = token_bytes
    for byte in range(256):
        token_bytes = bytes([byte])
        if token_bytes not in token_to_id:
            token_to_id[token_bytes] = len(vocab)
            vocab[len(vocab)] = token_bytes

    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size must be at least the initial vocabulary size ({len(vocab)}).")
    byte_token_ids = tuple(token_to_id[bytes([byte])] for byte in range(256))

    token_counts = _pretoken_counts(
        input_path,
        special_tokens,
        byte_token_ids=byte_token_ids,
        num_processes=num_processes,
    )
    pair_counts: dict[tuple[int, int], int] = {}
    pair_to_tokens: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for tokens, count in token_counts.items():
        _add_pair_counts(tokens, count, pair_counts, pair_to_tokens)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]))
        merged_token = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        merged_token_id = token_to_id.get(merged_token)
        if merged_token_id is None:
            merged_token_id = len(vocab)
            vocab[merged_token_id] = merged_token
            token_to_id[merged_token] = merged_token_id

        for tokens in list(pair_to_tokens.get(best_pair, ())):
            count = token_counts.pop(tokens)
            _remove_pair_counts(tokens, count, pair_counts, pair_to_tokens)

            merged_tokens = _merge_token_sequence(tokens, best_pair, merged_token_id)
            existing_count = token_counts.pop(merged_tokens, 0)
            if existing_count:
                _remove_pair_counts(merged_tokens, existing_count, pair_counts, pair_to_tokens)

            combined_count = count + existing_count
            token_counts[merged_tokens] = combined_count
            _add_pair_counts(merged_tokens, combined_count, pair_counts, pair_to_tokens)

    return vocab, merges
