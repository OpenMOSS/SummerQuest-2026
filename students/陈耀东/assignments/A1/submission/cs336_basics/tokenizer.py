from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex

from cs336_basics.bpe_train import PAT


# 类型别名：只是为了让函数签名和注释更清楚。
# TokenId 是 int，例如 0、1、50256。
# TokenBytes 是 bytes，例如 b"a"、b" the"、b"<|endoftext|>"。
TokenId = int
TokenBytes = bytes
Merge = tuple[TokenBytes, TokenBytes]
CONTRACTIONS = ("'s", "'d", "'m", "'t", "'ll", "'ve", "'re")
_BYTE_TOKENS = tuple(bytes([value]) for value in range(256))


class BPETokenizer:
    """byte-level BPE tokenizer。

    这也是学习脚手架：先搭清楚数据结构，再逐个补方法。

    Tokenizer 和 train_bpe 的区别：
    - train_bpe：从训练文本里“学出” vocab 和 merges。
    - Tokenizer：拿已经学好的 vocab 和 merges，把新文本 encode 成 token id，
      或把 token id decode 回文本。

    本类需要支持：
    - encode(text: str) -> list[int]
    - decode(ids: list[int]) -> str
    - encode_iterable(iterable: Iterable[str]) -> Iterator[int]
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        cache_capacity: int = 0,
    ) -> None:
        """从 vocab、merges、special_tokens 构造 tokenizer。

        Python 语法/数据结构提示：
        - self.xxx 表示“这个对象自己的属性”。
        - dict[int, bytes] 是 token id -> token bytes。
        - 但 encode 时更常需要反向查找：token bytes -> token id。
        - list[tuple[bytes, bytes]] 保留 merge 的训练顺序。
        - enumerate(merges) 可以同时得到 index 和 merge pair。

        你要建立的核心属性：
        - self.vocab：id 到 bytes。
        - self.token_to_id：bytes 到 id。
        - self.merges：按顺序保存的 merges。
        - self.merge_ranks：pair 到 rank，rank 越小表示越早 merge。
        - self.special_tokens：字符串形式的 special token。
        - self.special_token_bytes_to_id：special token 的 bytes 到 id。

        注意：
        - 如果 special token 不在 vocab 里，题目要求构造 tokenizer 时追加进 vocab。
        - special token 编码时必须保持完整，不能被拆开。
        - special token 重叠时，优先匹配更长的 token。
        """
        self.vocab: dict[TokenId, TokenBytes] = vocab
        self.token_to_id: dict[TokenBytes, TokenId] = {}
        for tokenid, tokenbyte in self.vocab.items():
            self.token_to_id[tokenbyte] = tokenid
        self.merges: list[Merge] = list(merges)
        self.merge_ranks: dict[Merge, int] = {}
        for rank, pair in enumerate(self.merges):
            self.merge_ranks[pair] = rank
        if special_tokens is None:
            self.special_tokens: list[str] = []
        else:
            self.special_tokens = list(special_tokens)
        if cache_capacity < 0:
            raise ValueError("cache_capacity 不能为负数")
        self.cache_capacity = cache_capacity
        self._encode_cache: dict[str, tuple[int, ...]] = {}
        self.special_token_bytes_to_id: dict[TokenBytes, TokenId] = {}
        for special in self.special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes not in self.token_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.token_to_id[special_bytes] = new_id
                self.vocab[new_id] = special_bytes
            self.special_token_bytes_to_id[special_bytes] = self.token_to_id[special_bytes]
        self.special_tokens_sorted: list[str] = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )
        if self.special_tokens_sorted:
            escaped_specials = [regex.escape(token) for token in self.special_tokens_sorted]
            self._special_pattern = regex.compile("(" + "|".join(escaped_specials) + ")")
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """从文件加载 tokenizer。

        这个方法是题目推荐接口之一，但当前测试主要通过 helper 直接传入 vocab/merges。
        如果后续要做 TinyStories 训练实验并保存/加载 tokenizer，再回来补这个方法。

        Python 语法提示：
        - @classmethod 表示这是“类方法”，第一个参数 cls 代表类本身。
        - 最后通常 return cls(vocab, merges, special_tokens)。
        """
        from cs336_basics.tokenizer_io import load_tokenizer_files

        vocab, merges = load_tokenizer_files(vocab_filepath, merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        result: list[int] = []

        # 1. 切分 special token 和普通文本
        segments = self._split_text_by_special_tokens(text)

        # 2. 逐段处理
        for piece, is_special in segments:
            if is_special:
                # 特殊 token 直接查 special_token_bytes_to_id
                piece_bytes = piece.encode("utf-8")
                result.append(self.special_token_bytes_to_id[piece_bytes])
            else:
                # 普通文本走普通编码逻辑
                result.extend(self._encode_ordinary_text(piece))

        return result

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码回字符串。

        高层流程：
        1. 对每个 token id，从 self.vocab 找到对应 bytes。
        2. 把所有 bytes 拼接成一个大 bytes。
        3. 最后整体 decode("utf-8", errors="replace")。

        为什么要“整体 decode”：
        - 一个 Unicode 字符可能由多个 UTF-8 byte 组成。
        - 如果逐 token 或逐 byte decode，可能把一个字符切碎导致错误。

        Python 语法提示：
        - b"".join(list_of_bytes) 可以拼接多个 bytes。
        - errors="replace" 表示遇到 malformed bytes 时用替换字符处理。
        """
        tokens_bytes: list[bytes] = []
        for id in ids:
            byte = self.vocab[id]
            tokens_bytes.append(byte)
        all_bytes = b"".join(tokens_bytes)
        return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码字符串 iterable，且结果与整段调用 ``encode`` 完全一致。

        chunk 是文件读取边界，不是 tokenizer 边界。每轮只输出已经确定不会被后续
        输入延长的完整 pre-token，并保留最后一个未决 pre-token 与可能跨 chunk 的
        special-token 前缀。这样既保持流式处理，也不会因为任意 chunk 切分改变结果。
        """
        pending = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("encode_iterable 的每个元素都必须是 str")
            pending += chunk
            safe_ids, pending = self._consume_stream_buffer(pending)
            yield from safe_ids

        if pending:
            yield from self.encode(pending)

    def _consume_stream_buffer(self, buffer: str) -> tuple[list[int], str]:
        """编码 buffer 中已经确定的前缀，并返回仍需等待后续输入的尾部。"""
        if not buffer:
            return [], ""

        partial_literal_length = self._longest_partial_literal_suffix(buffer)
        process_limit = len(buffer) - partial_literal_length
        encoded_ids: list[int] = []
        cursor = 0

        # 完整 special token 是硬边界，因此它之前的普通文本可以全部安全编码。
        if self._special_pattern is not None:
            for match in self._special_pattern.finditer(buffer, 0, process_limit):
                ordinary_text = buffer[cursor : match.start()]
                encoded_ids.extend(self._encode_ordinary_text(ordinary_text))
                special_bytes = match.group().encode("utf-8")
                encoded_ids.append(self.special_token_bytes_to_id[special_bytes])
                cursor = match.end()

        # 尾部普通文本的最后一个 pre-token 可能被下一个 chunk 延长，必须保留。
        trailing_text = buffer[cursor:process_limit]
        pretoken_matches = list(regex.finditer(PAT, trailing_text))
        if pretoken_matches:
            for match in pretoken_matches[:-1]:
                encoded_ids.extend(self._encode_pretoken(match.group()))
            unresolved_start = cursor + pretoken_matches[-1].start()
        else:
            unresolved_start = cursor

        return encoded_ids, buffer[unresolved_start:]

    def _longest_partial_literal_suffix(self, text: str) -> int:
        """返回 text 末尾与 special token 或缩写真前缀匹配的最长字符数。"""
        longest = 0
        protected_literals = (*self.special_tokens_sorted, *CONTRACTIONS)
        for literal in protected_literals:
            max_prefix_length = min(len(text), len(literal) - 1)
            for prefix_length in range(max_prefix_length, 0, -1):
                if prefix_length <= longest:
                    break
                if text.endswith(literal[:prefix_length]):
                    longest = prefix_length
                    break
        return longest

    def _split_text_by_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """把 text 切成普通片段和 special token 片段。

        返回值形状：
        - list[tuple[str, bool]]
        - 每个元素是 (piece, is_special)
        - is_special == True 表示 piece 是完整 special token。
        - is_special == False 表示 piece 是普通文本。

        关键要求：
        - special token 不能被拆开。
        - 如果 special tokens 有重叠，优先匹配更长的。
          例如 "<|endoftext|><|endoftext|>" 应先于 "<|endoftext|>" 匹配。

        实现方向提示：
        - 可以把 special_tokens 按长度从长到短排序。
        - 可以构造 regex pattern，并用捕获组保留分隔符。
        - 也可以手写从左到右扫描。
        """
        if not text:
            return []
        # 如果没有 special tokens，整段文本都是普通文本
        if not self.special_tokens:
            return [(text, False)]
        # 比如 "a<|end|>b" 会被 split 成 ["a", "<|end|>", "b"]
        assert self._special_pattern is not None
        parts = self._special_pattern.split(text)
        result: list[tuple[str, bool]] = []
        for part in parts:
            if not part:
                continue
            # 判断 part 是否是 special token
            is_special = part in self.special_tokens
            result.append((part, is_special))
        return result

    def _encode_ordinary_text(self, text: str) -> list[int]:
        ids: list[int] = []
        # 使用 finditer 找出文本里的所有 pre-tokens 并依次编码
        for match in regex.finditer(PAT, text):
            pretoken_text = match.group()
            ids.extend(self._encode_pretoken(pretoken_text))
        return ids

    def _encode_pretoken(self, pretoken_text: str) -> list[int]:
        cached_ids = self._encode_cache.get(pretoken_text)
        if cached_ids is not None:
            return list(cached_ids)

        # 1. 转为 UTF-8 bytes
        raw_bytes = pretoken_text.encode("utf-8")

        # 2. 拆成单字节的元组。注意：Python 中遍历 bytes 得到的是 int (0~255)，
        #    需要用 bytes([b]) 转回长度为 1 的 bytes。
        tokens = tuple(_BYTE_TOKENS[value] for value in raw_bytes)

        # 3. 应用 merges
        merged_tokens = self._apply_merges(tokens)

        # 4. 把合并后的 token bytes 查表转成对应的 id
        #    因为单字节和合并路径上的 token 都在 vocab 内，所以一定能查到
        token_ids = tuple(self.token_to_id[token] for token in merged_tokens)
        if self.cache_capacity > 0:
            if len(self._encode_cache) >= self.cache_capacity:
                self._encode_cache.clear()
            self._encode_cache[pretoken_text] = token_ids
        return list(token_ids)

    def _apply_merges(self, tokens: tuple[bytes, ...]) -> tuple[bytes, ...]:
        # 长度小于 2 无法形成 pair，不需要合并
        while len(tokens) >= 2:
            # 1. 找出当前所有相邻的 pair
            pairs = []
            for i in range(len(tokens) - 1):
                pairs.append((tokens[i], tokens[i + 1]))

            # 2. 找出其中 rank 最小（优先级最高）的 pair
            best_pair = None
            best_rank = float("inf")

            for pair in pairs:
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            # 3. 如果没有任何 pair 在 merge_ranks 里，说明合并结束了
            if best_pair is None:
                break

            # 4. 把序列中所有的 best_pair 合并掉
            first, second = best_pair
            merged_token = first + second
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = tuple(new_tokens)

        return tokens
