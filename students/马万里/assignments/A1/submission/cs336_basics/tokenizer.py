import heapq
import regex
from typing import Dict, List, Tuple, Iterator, Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        """
        Args:
            vocab: 词表映射，token ID -> token bytes
            merges: 训练得到的 BPE 合并规则，按创建顺序排列
            special_tokens: 用户指定的特殊 token，整体作为一个 token 处理
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)

        # 建立 pair -> merge 优先级（索引越小越早合并），用于高效合并
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

        self.special_tokens = list(special_tokens or [])

        # 建立 bytes -> ID 反向映射，编码时快速查找
        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # 自动将用户提供的 special tokens 添加到词表（如果不在）
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id
                next_id += 1

        # 快速查询 special token 对应的 ID
        self.special_token_to_id = {
            token: self.bytes_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

        # 按长度降序排列 special tokens，避免短 token 干扰长 token 匹配
        self.special_tokens_sorted = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        # 构造用于分割并保留 special token 的正则（捕获组保留分隔符）
        if self.special_tokens_sorted:
            pattern = "|".join(
                regex.escape(token)
                for token in self.special_tokens_sorted
            )
            self.special_regex = regex.compile(f"({pattern})")
        else:
            self.special_regex = None

    def _apply_merges(self, tokens: List[bytes]) -> List[bytes]:
        """
        对一个 pre-token 的字节列表按训练顺序应用 BPE merges。

        使用优先队列（最小堆）和链表结构实现高效合并：
        - 堆中存储 (merge_rank, 左节点索引)，rank 越小越优先
        - 每次弹出最小的有效 pair 执行合并，并更新相邻 pair
        复杂度 O(n log n)，远优于朴素 O(M*N)。
        """
        if not self.merges or len(tokens) <= 1:
            return tokens

        n = len(tokens)
        token_values = list(tokens)
        valid = [True] * n
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n)) + [-1]

        heap = []
        for i in range(n - 1):
            pair = (token_values[i], token_values[i + 1])
            rank = self.merge_rank.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        while heap:
            rank, i = heapq.heappop(heap)

            # 跳过已经失效的堆元素
            if not valid[i] or nxt[i] == -1 or not valid[nxt[i]]:
                continue
            j = nxt[i]
            a, b = self.merges[rank]
            if token_values[i] != a or token_values[j] != b:
                continue

            # 合并 i 和 j，使 i 成为合并后的节点，j 失效
            token_values[i] = a + b
            valid[j] = False

            old_prev = prev[i]
            old_next = nxt[j]
            nxt[i] = old_next
            if old_next != -1:
                prev[old_next] = i

            # 更新与新邻居形成的 pair 并压入堆
            if old_prev != -1 and valid[old_prev]:
                pair = (token_values[old_prev], token_values[i])
                new_rank = self.merge_rank.get(pair)
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, old_prev))

            if old_next != -1 and valid[old_next]:
                pair = (token_values[i], token_values[old_next])
                new_rank = self.merge_rank.get(pair)
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, i))

        # 重建最终 token 列表
        result = []
        i = 0
        while i != -1:
            result.append(token_values[i])
            i = nxt[i]
        return result

    def _encode_ordinary_text_gen(self, text: str) -> Iterator[int]:
        """
        对不含特殊 token 的普通文本进行编码，逐 token 产出。
        使用 GPT-2 正则预分词，每个 pre-token 应用 merges。
        """
        for match in regex.finditer(PAT, text):
            token_bytes = match.group(0).encode("utf-8")
            tokens = [bytes([b]) for b in token_bytes]
            merged = self._apply_merges(tokens)
            for token in merged:
                yield self.bytes_to_id[token]

    def _encode_gen(self, text: str) -> Iterator[int]:
        """
        对整个文本编码，逐 token 产出。
        若存在特殊 token，先按 special_regex 分割，区分处理。
        """
        if not self.special_tokens_sorted:
            yield from self._encode_ordinary_text_gen(text)
            return

        for part in self.special_regex.split(text):
            if not part:
                continue
            special_id = self.special_token_to_id.get(part)
            if special_id is not None:
                # 特殊 token 直接映射为 ID，不参与 merges
                yield special_id
            else:
                yield from self._encode_ordinary_text_gen(part)

    def encode(self, text: str) -> List[int]:
        return list(self._encode_gen(text))

    def decode(self, ids: List[int]) -> str:
        """
        将 token ID 列表解码为 UTF-8 字符串。
        将所有 token 的 bytes 拼接后一次性解码，非法字节替换为 U+FFFD。
        """
        byte_parts = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"Invalid token id: {token_id}")
            byte_parts.append(self.vocab[token_id])

        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式编码：逐块处理输入，按换行符切分完整行，避免 token 跨块。
        内存占用与输入总大小无关。
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            last_newline = buffer.rfind("\n")
            if last_newline != -1:
                # 处理换行符之前的完整行
                to_process = buffer[: last_newline + 1]
                buffer = buffer[last_newline + 1 :]
                yield from self._encode_gen(to_process)
        # 处理剩余不完整的行
        if buffer:
            yield from self._encode_gen(buffer)