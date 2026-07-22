"""Byte-level BPE tokenization utilities.

The tokenizer deliberately stores tokens as raw ``bytes``.  The printable
byte-to-Unicode representation used by GPT-2 is only a serialization format;
it is converted back to bytes by :meth:`Tokenizer.from_files`.
"""

from __future__ import annotations

import heapq
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex


# This is the pre-tokenization expression used by GPT-2.  In particular, the
# two whitespace alternatives and their order are significant.
GPT2_PRETOKEN_PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def _normalize_special_tokens(special_tokens: Iterable[str] | None) -> tuple[str, ...]:
    """Return unique special tokens in longest-match order."""

    if special_tokens is None:
        return ()

    unique: dict[str, None] = {}
    for token in special_tokens:
        if not isinstance(token, str):
            raise TypeError("special tokens must be strings")
        if token == "":
            raise ValueError("the empty string cannot be a special token")
        unique.setdefault(token, None)
    return tuple(sorted(unique, key=lambda token: (-len(token), token)))


def _yield_normal_pretokens(text: str) -> Iterator[tuple[str, bool]]:
    for match in GPT2_PRETOKEN_PATTERN.finditer(text):
        yield match.group(), False


def _consume_text_buffer(
    text: str,
    special_tokens_by_initial: dict[str, tuple[str, ...]],
    *,
    final: bool,
) -> Iterator[tuple[str, bool]]:
    """Tokenize the settled part of ``text`` and return its unsettled suffix.

    The generator's return value is the suffix that needs more input.  Holding
    back the final ordinary pre-token is what makes chunk boundaries invisible
    to the GPT-2 regular expression.  If the input ends in a prefix of a
    special token, the ordinary pre-token containing that prefix is retained as
    well.
    """

    text_length = len(text)
    scan_position = 0
    normal_start = 0
    partial_special_start: int | None = None

    while scan_position < text_length:
        candidates = special_tokens_by_initial.get(text[scan_position])
        if not candidates:
            scan_position += 1
            continue

        remaining_length = text_length - scan_position
        matched_special: str | None = None
        has_longer_partial_match = False

        # Candidates are ordered longest first, which implements longest-match
        # semantics for overlapping special tokens.
        for special_token in candidates:
            special_length = len(special_token)
            if special_length <= remaining_length:
                if matched_special is None and text.startswith(special_token, scan_position):
                    matched_special = special_token
            elif not final:
                remaining_text = text[scan_position:]
                if special_token.startswith(remaining_text):
                    has_longer_partial_match = True

        # A complete short token cannot be committed when the current suffix
        # could still become a longer special token after another chunk.
        if has_longer_partial_match:
            partial_special_start = scan_position
            break

        if matched_special is None:
            scan_position += 1
            continue

        yield from _yield_normal_pretokens(text[normal_start:scan_position])
        yield matched_special, True
        scan_position += len(matched_special)
        normal_start = scan_position

    trailing_text = text[normal_start:]
    if final:
        yield from _yield_normal_pretokens(trailing_text)
        return ""

    if not trailing_text:
        return ""

    partial_offset = None
    partial_context_offset = None
    if partial_special_start is not None:
        partial_offset = partial_special_start - normal_start
        partial_context_offset = partial_offset
        # If the candidate special begins immediately after whitespace, that
        # whitespace must remain unsettled too.  Once the special is confirmed,
        # it becomes an end-of-string boundary for the normal segment, which
        # can change ``\s+(?!\S)`` grouping (for example ``"\n abc"`` when
        # ``"abc"`` is special).
        while partial_context_offset > 0 and trailing_text[partial_context_offset - 1].isspace():
            partial_context_offset -= 1

    last_match_start: int | None = None
    partial_match_start: int | None = None
    for match in GPT2_PRETOKEN_PATTERN.finditer(trailing_text):
        last_match_start = match.start()
        if partial_context_offset is not None and match.start() <= partial_context_offset < match.end():
            partial_match_start = match.start()

    # The expression is exhaustive for non-empty text.  Keeping the whole
    # buffer is the safest fallback if that invariant is ever changed.
    if last_match_start is None:
        return trailing_text

    keep_from = partial_match_start if partial_match_start is not None else last_match_start
    # Re-run the expression against the *full* trailing text and emit matches
    # before the retained suffix.  Running it on ``trailing_text[:keep_from]``
    # would introduce an artificial end-of-string and can change the
    # ``\s+(?!\S)`` alternative (for example, ``"\n\t>"``).
    for match in GPT2_PRETOKEN_PATTERN.finditer(trailing_text):
        if match.end() > keep_from:
            break
        yield match.group(), False
    return trailing_text[keep_from:]


def iter_pretokens(
    chunks: Iterable[str],
    special_tokens: Iterable[str] | None = None,
) -> Iterator[tuple[str, bool]]:
    """Yield ``(piece, is_special)`` pairs independently of chunk boundaries.

    Only the final not-yet-settled pre-token (and, when necessary, a partial
    special token) is retained, so this works for inputs much larger than RAM.
    """

    normalized_specials = _normalize_special_tokens(special_tokens)
    specials_by_initial: dict[str, list[str]] = {}
    for token in normalized_specials:
        specials_by_initial.setdefault(token[0], []).append(token)
    frozen_specials_by_initial = {key: tuple(value) for key, value in specials_by_initial.items()}

    buffer = ""
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError("Tokenizer input chunks must be strings")
        if not chunk:
            continue
        buffer += chunk
        buffer = yield from _consume_text_buffer(buffer, frozen_specials_by_initial, final=False)

    if buffer:
        yield from _consume_text_buffer(buffer, frozen_specials_by_initial, final=True)


def _gpt2_byte_decoder() -> dict[str, int]:
    """Map GPT-2's printable Unicode alphabet back to byte values."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values[:]
    offset = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            codepoints.append(256 + offset)
            offset += 1
    return {chr(codepoint): byte_value for byte_value, codepoint in zip(byte_values, codepoints, strict=True)}


def _gpt2_byte_encoder() -> dict[int, str]:
    return {byte_value: character for character, byte_value in _gpt2_byte_decoder().items()}


def _deserialize_token(token: str, byte_decoder: dict[str, int]) -> bytes:
    try:
        return bytes(byte_decoder[character] for character in token)
    except KeyError:
        # This fallback makes hand-written vocabularies with literal Unicode
        # special tokens usable while preserving GPT-2 file compatibility.
        return token.encode("utf-8")


def _serialize_token(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[byte_value] for byte_value in token)


class Tokenizer:
    """A deterministic byte-level BPE tokenizer."""

    _MAX_LOCAL_CACHE_SIZE = 256

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = {int(token_id): bytes(token) for token_id, token in vocab.items()}
        self.merges = [(bytes(left), bytes(right)) for left, right in merges]
        self.special_tokens = _normalize_special_tokens(special_tokens)

        token_to_id: dict[bytes, int] = {}
        for token_id in sorted(self.vocab):
            token_to_id.setdefault(self.vocab[token_id], token_id)

        next_token_id = max(self.vocab, default=-1) + 1
        self._special_token_to_id: dict[str, int] = {}
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            token_id = token_to_id.get(token_bytes)
            if token_id is None:
                token_id = next_token_id
                next_token_id += 1
                self.vocab[token_id] = token_bytes
                token_to_id[token_bytes] = token_id
            self._special_token_to_id[special_token] = token_id

        self._token_to_id = token_to_id
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        self._single_byte_tokens: list[bytes] = []
        for byte_value in range(256):
            token = bytes((byte_value,))
            if token not in self._token_to_id:
                raise ValueError(f"vocabulary is missing the base byte token {byte_value}")
            self._single_byte_tokens.append(token)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load GPT-2-style ``vocab.json`` and ``merges.txt`` files.

        If ``special_tokens`` is omitted and a sibling
        ``tokenizer_config.json`` produced by :meth:`save` exists, special
        tokens are restored from that file automatically.
        """

        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        if special_tokens is None:
            config_path = vocab_path.parent / "tokenizer_config.json"
            if config_path.is_file():
                with config_path.open(encoding="utf-8") as config_file:
                    tokenizer_config = json.load(config_file)
                configured_specials = tokenizer_config.get("special_tokens")
                if configured_specials is not None:
                    special_tokens = [str(token) for token in configured_specials]

        byte_decoder = _gpt2_byte_decoder()
        with vocab_path.open(encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)

        vocab: dict[int, bytes]
        if all(isinstance(value, int) for value in serialized_vocab.values()):
            vocab = {
                int(token_id): _deserialize_token(str(serialized_token), byte_decoder)
                for serialized_token, token_id in serialized_vocab.items()
            }
        elif all(str(key).lstrip("-").isdigit() for key in serialized_vocab):
            # A small convenience for ID-to-token JSON files produced by local
            # experiment scripts.
            vocab = {
                int(token_id): _deserialize_token(str(serialized_token), byte_decoder)
                for token_id, serialized_token in serialized_vocab.items()
            }
        else:
            raise ValueError("unsupported vocabulary JSON format")

        merges: list[tuple[bytes, bytes]] = []
        with merges_path.open(encoding="utf-8") as merges_file:
            for line in merges_file:
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith("#"):
                    continue
                pieces = stripped_line.split()
                if len(pieces) != 2:
                    continue
                merges.append(
                    (
                        _deserialize_token(pieces[0], byte_decoder),
                        _deserialize_token(pieces[1], byte_decoder),
                    )
                )

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @classmethod
    def from_directory(
        cls,
        directory: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load the three-file tokenizer layout written by :meth:`save`."""

        directory_path = Path(directory)
        return cls.from_files(
            directory_path / "vocab.json",
            directory_path / "merges.txt",
            special_tokens=special_tokens,
        )

    def save(self, output_dir: str | os.PathLike[str]) -> dict[str, Path]:
        """Persist this tokenizer in a deterministic, byte-lossless format.

        The vocabulary and merges use GPT-2's printable byte alphabet, so the
        resulting files are also consumable by standard GPT-2-style tooling.
        A small config file records special-token semantics that cannot be
        inferred from vocabulary bytes alone.
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        vocab_path = output_path / "vocab.json"
        merges_path = output_path / "merges.txt"
        config_path = output_path / "tokenizer_config.json"

        byte_encoder = _gpt2_byte_encoder()
        serialized_vocab: dict[str, int] = {}
        for token_id in sorted(self.vocab):
            serialized_token = _serialize_token(self.vocab[token_id], byte_encoder)
            if serialized_token in serialized_vocab:
                raise ValueError("cannot serialize a vocabulary containing duplicate token bytes")
            serialized_vocab[serialized_token] = token_id

        with vocab_path.open("w", encoding="utf-8", newline="\n") as vocab_file:
            json.dump(serialized_vocab, vocab_file, ensure_ascii=False, indent=2)
            vocab_file.write("\n")

        with merges_path.open("w", encoding="utf-8", newline="\n") as merges_file:
            merges_file.write("#version: 0.2\n")
            for left, right in self.merges:
                merges_file.write(f"{_serialize_token(left, byte_encoder)} {_serialize_token(right, byte_encoder)}\n")

        config = {
            "format": "cs336_byte_bpe_v1",
            "special_tokens": list(self.special_tokens),
            "vocab_size": len(self.vocab),
        }
        with config_path.open("w", encoding="utf-8", newline="\n") as config_file:
            json.dump(config, config_file, ensure_ascii=False, indent=2, sort_keys=True)
            config_file.write("\n")

        return {"vocab": vocab_path, "merges": merges_path, "config": config_path}

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        raw_bytes = pretoken.encode("utf-8")
        if not raw_bytes:
            return ()

        token_values = [self._single_byte_tokens[byte_value] for byte_value in raw_bytes]
        token_count = len(token_values)
        if token_count == 1:
            return (self._token_to_id[token_values[0]],)

        previous = [index - 1 for index in range(token_count)]
        following = [index + 1 for index in range(token_count)]
        following[-1] = -1
        alive = [True] * token_count
        candidate_merges: list[tuple[int, int, int]] = []

        def add_candidate(left_index: int, right_index: int) -> None:
            if left_index < 0 or right_index < 0:
                return
            rank = self._merge_ranks.get((token_values[left_index], token_values[right_index]))
            if rank is not None:
                heapq.heappush(candidate_merges, (rank, left_index, right_index))

        for left_index in range(token_count - 1):
            add_candidate(left_index, left_index + 1)

        while candidate_merges:
            rank, left_index, right_index = heapq.heappop(candidate_merges)
            if not alive[left_index] or not alive[right_index] or following[left_index] != right_index:
                continue
            pair = (token_values[left_index], token_values[right_index])
            if self._merge_ranks.get(pair) != rank:
                continue

            token_values[left_index] = pair[0] + pair[1]
            alive[right_index] = False
            right_neighbor = following[right_index]
            following[left_index] = right_neighbor
            if right_neighbor >= 0:
                previous[right_neighbor] = left_index

            left_neighbor = previous[left_index]
            add_candidate(left_neighbor, left_index)
            add_candidate(left_index, right_neighbor)

        encoded_ids: list[int] = []
        index = 0
        while index >= 0:
            encoded_ids.append(self._token_to_id[token_values[index]])
            index = following[index]
        return tuple(encoded_ids)

    def _encode_chunks(self, chunks: Iterable[str]) -> Iterator[int]:
        # A tiny cache captures common words without retaining an unbounded set
        # of pre-tokens during streaming encoding.
        cache: dict[str, tuple[int, ...]] = {}
        for piece, is_special in iter_pretokens(chunks, self.special_tokens):
            if is_special:
                yield self._special_token_to_id[piece]
                continue

            encoded_piece = cache.get(piece)
            if encoded_piece is None:
                encoded_piece = self._encode_pretoken(piece)
                if len(cache) >= self._MAX_LOCAL_CACHE_SIZE:
                    cache.clear()
                cache[piece] = encoded_piece
            yield from encoded_piece

    def encode(self, text: str) -> list[int]:
        """Encode one string into token IDs."""

        if not isinstance(text, str):
            raise TypeError("Tokenizer.encode expects a string")
        return list(self._encode_chunks((text,)))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode string chunks without changing tokenization at boundaries."""

        yield from self._encode_chunks(iterable)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs, replacing malformed UTF-8 with U+FFFD."""

        decoded_bytes = b"".join(self.vocab[int(token_id)] for token_id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")
