import json
from collections.abc import Iterable, Iterator

import regex as re


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, vocab: dict[int, bytes], merges, special_tokens: list[str] | None = None
    ) -> None:
        """
        Construct a tokenizer from a given  vocabulary, list of merges, and (optionally) a list of special tokens.
        This function should accept the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.reverse_vocab = dict(zip(vocab.values(), vocab.keys()))
        self.merges = merges
        self.merge_ranks = {merge: rank for rank, merge in enumerate(merges)}
        self.special_tokens = None
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=lambda x: (-len(x), x))
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.reverse_vocab:
                    token_id = len(self.vocab)
                    self.vocab[token_id] = token_bytes
                    self.reverse_vocab[token_bytes] = token_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        vocab = {}
        merges: list[tuple[bytes, bytes]] = []
        if vocab_filepath:
            with open(vocab_filepath, encoding="utf-8") as f:
                data = json.load(f)
                vocab = cls._load_vocab_payload(data)
        if merges_filepath:
            with open(merges_filepath, encoding="utf-8") as f:
                merges = cls._load_merges_payload(f.read())
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _load_vocab_payload(data) -> dict[int, bytes]:
        if isinstance(data, dict) and all(
            isinstance(token_id, str)
            and token_id.isdigit()
            and isinstance(token_hex, str)
            for token_id, token_hex in data.items()
        ):
            return {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in data.items()}

        if isinstance(data, dict):
            return {int(token_id): token.encode("utf-8") for token, token_id in data.items()}

        raise ValueError("Unsupported vocab file format")

    @staticmethod
    def _load_merges_payload(text: str) -> list[tuple[bytes, bytes]]:
        stripped = text.strip()
        if not stripped:
            return []

        if stripped.startswith("["):
            data = json.loads(stripped)
            return [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in data]

        merges: list[tuple[bytes, bytes]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token1, token2 = line.split()
            merges.append((token1.encode("utf-8"), token2.encode("utf-8")))
        return merges

    def pre_tokenization(self, input_str: str) -> list[list[list[bytes]]]:
        # handle special_tokens
        # parts = [input_str]
        if self.special_tokens:
            pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            parts = re.split(f"({pattern})", input_str)
        else:
            parts = [input_str]

        res = [[[]]]
        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                res.append([[part.encode("utf-8")]])
            else:
                res.append(
                    [
                        [bytes([byte]) for byte in word.group(0).encode("utf-8")]
                        for word in re.finditer(self.PAT, part)
                    ]
                )
        return res

    def encode(self, text: str) -> list[int]:
        sentences = self.pre_tokenization(text)
        results: list[int] = []
        for sentence in sentences:
            for word in sentence:
                results.extend(self._encode_word(word))
        return results

    def _encode_word(self, word: list[bytes]) -> list[int]:
        if not word:
            return []

        tokens = tuple(word)
        while len(tokens) > 1:
            best_pair = None
            best_rank = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            tokens = self._merge_pair(tokens, best_pair)

        return [self.reverse_vocab[token] for token in tokens]

    @staticmethod
    def _merge_pair(tokens: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        merged_token = pair_to_merge[0] + pair_to_merge[1]
        merged_tokens: list[bytes] = []
        index = 0
        while index < len(tokens):
            if index < len(tokens) - 1 and (tokens[index], tokens[index + 1]) == pair_to_merge:
                merged_tokens.append(merged_token)
                index += 2
            else:
                merged_tokens.append(tokens[index])
                index += 1
        return tuple(merged_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of  strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            # 假设 self.encode 返回 list[int]
            yield from self.encode(text)  # 逐个 yield，不累积

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        data = b"".join(self.vocab[_] for _ in ids)
        return data.decode("utf-8", errors="replace")
