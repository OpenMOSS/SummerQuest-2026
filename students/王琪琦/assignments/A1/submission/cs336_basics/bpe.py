from __future__ import annotations

import os
from collections import Counter, defaultdict

import regex


GPT2_PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _count_pretokens(
    input_path: str | os.PathLike[str], special_tokens: list[str]
) -> Counter[tuple[int, ...]]:
    """Count UTF-8 byte sequences after GPT-2-style pre-tokenization."""
    with open(input_path, encoding="utf-8") as corpus_file:
        text = corpus_file.read()

    if special_tokens:
        # Special tokens are document boundaries: exclude them before ordinary
        # pre-tokenization so pairs can never cross or include a boundary.
        alternatives = sorted(set(special_tokens), key=len, reverse=True)
        special_pattern = regex.compile(
            "|".join(regex.escape(token) for token in alternatives)
        )
        text_segments = special_pattern.split(text)
    else:
        text_segments = [text]

    counts: Counter[tuple[int, ...]] = Counter()
    for segment in text_segments:
        counts.update(
            tuple(match.group().encode("utf-8"))
            for match in GPT2_PRETOKEN_PATTERN.finditer(segment)
        )
    return counts


def _pair_multiplicities(tokens: tuple[int, ...]) -> Counter[tuple[int, int]]:
    return Counter(zip(tokens, tokens[1:]))


def _merge_pair(
    tokens: tuple[int, ...], pair: tuple[int, int], new_token_id: int
) -> tuple[int, ...]:
    """Replace every non-overlapping occurrence of pair from left to right."""
    merged: list[int] = []
    index = 0
    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == pair[0]
            and tokens[index + 1] == pair[1]
        ):
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary with deterministic merge ordering."""
    unique_special_tokens = list(dict.fromkeys(special_tokens))
    minimum_vocab_size = 256 + len(unique_special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {minimum_vocab_size} for the byte "
            "vocabulary and requested special tokens"
        )

    vocab: dict[int, bytes] = {byte_value: bytes([byte_value]) for byte_value in range(256)}
    for token in unique_special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    pretoken_counts = _count_pretokens(input_path, unique_special_tokens)
    word_sequences = list(pretoken_counts)
    word_frequencies = [pretoken_counts[word] for word in word_sequences]

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for word_id, (tokens, frequency) in enumerate(zip(word_sequences, word_frequencies)):
        for pair, occurrences in _pair_multiplicities(tokens).items():
            pair_counts[pair] += occurrences * frequency
            pair_to_words[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        new_token_id = len(vocab)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        affected_word_ids = tuple(pair_to_words[best_pair])
        for word_id in affected_word_ids:
            old_tokens = word_sequences[word_id]
            frequency = word_frequencies[word_id]

            for pair, occurrences in _pair_multiplicities(old_tokens).items():
                pair_counts[pair] -= occurrences * frequency
                pair_to_words[pair].discard(word_id)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                    del pair_to_words[pair]

            new_tokens = _merge_pair(old_tokens, best_pair, new_token_id)
            word_sequences[word_id] = new_tokens
            for pair, occurrences in _pair_multiplicities(new_tokens).items():
                pair_counts[pair] += occurrences * frequency
                pair_to_words[pair].add(word_id)

    return vocab, merges
