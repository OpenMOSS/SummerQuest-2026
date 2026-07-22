import regex as re
from collections.abc import Iterable, Iterator

from cs336_basics.bpe import merge_pretoken

PRETOKEN_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        # 降序排列，较长 special token 排在前面，使 <x><x> 与 <x> 重叠时优先匹配前者
        self.special_tokens = sorted(special_tokens or [], key=lambda token: len(token), reverse=True)
        next_id = max(self.vocab, default=-1) + 1
        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.token_to_id:
                    self.vocab[next_id] = token_bytes
                    self.token_to_id[token_bytes] = next_id
                    next_id += 1
        self.special_token_to_id = {token: self.token_to_id[token.encode()] for token in self.special_tokens}

        self.special_pattern = (
            re.compile(
                "|".join(
                    # re.escape 防止 special token 中的 `|`, `*` 被当成正则语法
                    re.escape(token)
                    for token in self.special_tokens
                )
            )
            if self.special_tokens else None
        )

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        tokens: tuple[bytes, ...] = tuple(bytes([b]) for b in pretoken)

        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)
                # rank 小表示训练时更早确定的合并规则，后续 merge 以早期生成的 token 为输入，所以必须按依赖顺序重放。
                if rank is not None and (
                    best_rank is None or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                return [
                    self.token_to_id[token] for token in tokens
                ]
            tokens = merge_pretoken(tokens, best_pair)


    def _encode_ordinary(self, text: str) -> list[int]:
        """不考虑 special token 时的分词逻辑"""
        ids = []

        # 流式预分词
        for match in PRETOKEN_PATTERN.finditer(text):
            pretoken = match.group().encode("utf-8")
            ids.extend(self._encode_pretoken(pretoken))

        return ids

    def encode(self, text: str) -> list[int]:
        """处理 special token"""
        if self.special_pattern is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        cursor = 0

        for match in self.special_pattern.finditer(text):
            ordinary_text = text[cursor:match.start()]
            ids.extend(self._encode_ordinary(ordinary_text))

            special_token = match.group()
            ids.append(self.special_token_to_id[special_token])

            cursor = match.end()

        ids.extend(self._encode_ordinary(text[cursor:]))
        return ids

    def decode(self, ids: list[int]) -> str:
        # 必须先拼接全部 bytes，再进行一次 UTF-8 decode，因为但个 byte token 可能不是合法的独立 UTF-8 字符
        token_bytes = b"".join(
            self.vocab[token_id] for token_id in ids
        )

        return token_bytes.decode("utf-8", errors="replace")


    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

if __name__ == "__main__":
    vocab = {
        0: b" ", 1: b"a", 2: b"c", 3: b"e",
        4: b"h", 5: b"t", 6: b"th", 7: b" c",
        8: b" a", 9: b"the", 10: b" at",
    }
    merges = [
        (b"t", b"h"),
        (b" ", b"c"),
        (b" ", b"a"),
        (b"th", b"e"),
        (b" a", b"t")
    ]

    assert Tokenizer(vocab, merges).encode("the cat ate") == [9, 7, 1, 5, 10, 3]

    base_vocab = {i: bytes([i]) for i in range(256)}

    tokenizer = Tokenizer(
        base_vocab, [], ["<x>", "<x><x>"]
    )

    print(tokenizer.decode(tokenizer.encode("he<x>llo")))
    assert tokenizer.decode(tokenizer.encode("你<x>好")) == "你<x>好"

    double_id = tokenizer.special_token_to_id["<x><x>"]
    assert tokenizer.encode("<x><x>") == [double_id]

    tokenizer = Tokenizer({0: b"\xe4"}, [])
    assert tokenizer.decode([0]) == "\ufffd"