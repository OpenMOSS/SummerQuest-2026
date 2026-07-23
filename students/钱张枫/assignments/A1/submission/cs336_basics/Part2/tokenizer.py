from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import regex as re


GPT2_PRETOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# merges 和词表项都用原始 bytes 表示。这样 tokenizer 不依赖 GPT-2
# vocab.json 这类仅用于展示和序列化的 Unicode 字节映射。
Pair = tuple[bytes, bytes]


class Tokenizer:
    """使用 GPT-2 预分词规则的 byte-level BPE tokenizer。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # 复制调用方传入的数据，避免追加缺失 special token 时修改训练产物或测试 fixture。
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        # special token 在编码时必须作为原子 token。如果调用方传入的 special token
        # 不在训练得到的词表里，就追加到现有词表末尾，保证原始 token ID 不变。
        existing_token_bytes = set(self.vocab.values())
        next_token_id = max(self.vocab.keys(), default=-1) + 1
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in existing_token_bytes:
                self.vocab[next_token_id] = special_token_bytes
                existing_token_bytes.add(special_token_bytes)
                next_token_id += 1

        # 编码需要 bytes -> token id 的反查表；BPE merge 选择需要 pair -> rank。
        # rank 越小表示该 merge 越早在训练中学到，编码时优先级越高。
        self._token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._pretoken_pattern = re.compile(GPT2_PRETOKENIZATION_PATTERN)
        self._special_pattern = self._compile_special_pattern(self.special_tokens)
        self._max_special_token_length = max((len(token) for token in self.special_tokens), default=0)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        # 在 GPT-2 预分词之前先切出 special token，确保它们不会被拆分，
        # 也不会和相邻普通文本发生 BPE 合并。
        for segment, is_special_token in self._split_text(text):
            if is_special_token:
                ids.append(self._token_to_id[segment.encode("utf-8")])
                continue

            for match in self._pretoken_pattern.finditer(segment):
                ids.extend(self._encode_pretoken(match.group(0).encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 把 iterable 当作连续文本流处理。chunk 边界可能切在单词、连续空白或
        # special token 中间，因此只输出确定不依赖后续输入的安全前缀。
        buffer = ""
        for text in iterable:
            buffer += text
            safe_prefix_length = self._safe_prefix_length(buffer)
            if safe_prefix_length == 0:
                continue

            safe_prefix = buffer[:safe_prefix_length]
            emit_text, tail_text = self._split_flushable_prefix(safe_prefix)
            if emit_text:
                yield from self.encode(emit_text)
            buffer = tail_text + buffer[safe_prefix_length:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        # byte-level tokenizer 先还原并拼接 bytes，再统一做一次 UTF-8 解码。
        # 对非法 UTF-8 字节序列使用替换字符，符合测试和作业要求。
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def _split_text(self, text: str) -> Iterator[tuple[str, bool]]:
        if self._special_pattern is None:
            if text:
                yield text, False
            return

        current_index = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > current_index:
                yield text[current_index : match.start()], False
            yield match.group(0), True
            current_index = match.end()

        if current_index < len(text):
            yield text[current_index:], False

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        # byte-level BPE 从单字节开始；只要基础 0..255 字节词表存在，
        # 任意 UTF-8 输入字节都可以被表示。
        parts = [bytes([byte]) for byte in pretoken]

        while len(parts) > 1:
            best_pair: Pair | None = None
            best_pair_rank = len(self._merge_ranks)

            # 每轮 BPE 选择当前序列中 rank 最小的相邻 pair，相当于按训练时
            # merge 的生成顺序在当前 pretoken 上重放合并规则。
            for index in range(len(parts) - 1):
                pair = (parts[index], parts[index + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_pair_rank:
                    best_pair = pair
                    best_pair_rank = rank

            if best_pair is None:
                break

            left, right = best_pair
            merged_parts: list[bytes] = []
            index = 0
            # 先合并当前最佳 pair 的所有非重叠出现，再重新计算下一轮最佳 pair。
            while index < len(parts):
                if index < len(parts) - 1 and parts[index] == left and parts[index + 1] == right:
                    merged_parts.append(left + right)
                    index += 2
                else:
                    merged_parts.append(parts[index])
                    index += 1
            parts = merged_parts

        return [self._token_to_id[token] for token in parts]

    def _safe_prefix_length(self, text: str) -> int:
        if self._max_special_token_length == 0:
            return len(text)

        # 保留足够长的尾部字符，用来识别可能从当前 buffer 尾部开始、
        # 但要到后续 chunk 才结束的 special token。
        safe_prefix_length = max(0, len(text) - self._max_special_token_length + 1)
        if self._special_pattern is None:
            return safe_prefix_length

        # 如果候选边界落在已经匹配到的 special token 内部，就把边界左移，
        # 确保 special token 仍然作为原子整体处理。
        for match in self._special_pattern.finditer(text):
            if match.start() < safe_prefix_length < match.end():
                return match.start()
        return safe_prefix_length

    def _split_flushable_prefix(self, text: str) -> tuple[str, str]:
        # special token 安全边界之外，前缀仍可能结束在普通 GPT-2 pretoken 内部。
        # 因此保留最后一个普通 pretoken，保证 iterable 编码结果与整段字符串编码一致。
        last_segment_start = 0
        last_segment_end = len(text)
        last_segment_is_special = False

        if self._special_pattern is not None:
            current_index = 0
            for match in self._special_pattern.finditer(text):
                if match.start() > current_index:
                    last_segment_start = current_index
                    last_segment_end = match.start()
                    last_segment_is_special = False

                last_segment_start = match.start()
                last_segment_end = match.end()
                last_segment_is_special = True
                current_index = match.end()

            if current_index < len(text):
                last_segment_start = current_index
                last_segment_end = len(text)
                last_segment_is_special = False

        if last_segment_is_special:
            return text, ""

        last_segment = text[last_segment_start:last_segment_end]
        last_match: re.Match[str] | None = None
        for match in self._pretoken_pattern.finditer(last_segment):
            last_match = match

        if last_match is not None and last_match.end() == len(last_segment):
            tail_start = last_segment_start + last_match.start()
            return text[:tail_start], text[tail_start:]

        return text, ""

    @staticmethod
    def _compile_special_pattern(special_tokens: Sequence[str]) -> re.Pattern[str] | None:
        if not special_tokens:
            return None
        # 按长度降序匹配，确保重叠 special token 优先选择更具体的长 token，
        # 例如先匹配 "<A><A>"，再匹配 "<A>"。
        escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        return re.compile("|".join(escaped_tokens))


def load_vocab(vocab_filepath: str) -> dict[int, bytes]:
    payload = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))

    # 本项目训练脚本输出的产物以 JSON object key 保存 token id，
    # 并用十六进制字段保存 token bytes：{"123": {"hex": "...", ...}}。
    if isinstance(payload, dict) and all(str(key).isdigit() for key in payload):
        vocab: dict[int, bytes] = {}
        for token_id_string, token_payload in payload.items():
            vocab[int(token_id_string)] = _decode_serialized_vocab_token(token_payload)
        return vocab

    # GPT-2 fixture 以“展示字符串 -> token id”保存词表。
    # 这里把每个展示字符还原成原始字节值。
    if isinstance(payload, dict) and all(isinstance(token_id, int) for token_id in payload.values()):
        byte_decoder = _gpt2_byte_decoder()
        return {
            token_id: bytes(byte_decoder[character] for character in token_string)
            for token_string, token_id in payload.items()
        }

    raise ValueError(f"Unsupported vocab serialization format: {vocab_filepath}")


def load_merges(merges_filepath: str) -> list[Pair]:
    path = Path(merges_filepath)
    text = path.read_text(encoding="utf-8")

    # 本项目训练脚本输出的 merges JSON 使用 left_hex/right_hex 字段。
    if path.suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported merges serialization format: {merges_filepath}")
        return [_decode_serialized_merge(merge_payload) for merge_payload in payload]

    # GPT-2 merges 文件是纯文本 pair，使用与 GPT-2 vocab.json 相同的展示 Unicode 空间。
    byte_decoder = _gpt2_byte_decoder()
    merges: list[Pair] = []
    for line in text.splitlines():
        cleaned_line = line.rstrip()
        if not cleaned_line or cleaned_line.startswith("#"):
            continue
        pieces = cleaned_line.split(" ")
        if len(pieces) != 2:
            continue
        left, right = pieces
        merges.append(
            (
                bytes(byte_decoder[character] for character in left),
                bytes(byte_decoder[character] for character in right),
            )
        )
    return merges


def _decode_serialized_vocab_token(token_payload: Any) -> bytes:
    if isinstance(token_payload, dict) and isinstance(token_payload.get("hex"), str):
        return bytes.fromhex(token_payload["hex"])
    if isinstance(token_payload, str):
        return bytes.fromhex(token_payload)
    raise ValueError(f"Unsupported vocab token payload: {token_payload!r}")


def _decode_serialized_merge(merge_payload: Any) -> Pair:
    if (
        isinstance(merge_payload, dict)
        and isinstance(merge_payload.get("left_hex"), str)
        and isinstance(merge_payload.get("right_hex"), str)
    ):
        return bytes.fromhex(merge_payload["left_hex"]), bytes.fromhex(merge_payload["right_hex"])

    if (
        isinstance(merge_payload, (list, tuple))
        and len(merge_payload) == 2
        and all(isinstance(part, str) for part in merge_payload)
    ):
        return bytes.fromhex(merge_payload[0]), bytes.fromhex(merge_payload[1])

    raise ValueError(f"Unsupported merge payload: {merge_payload!r}")


def _gpt2_byte_decoder() -> dict[str, int]:
    # GPT-2 为了写入 JSON/text 文件，会把字节映射到可打印 Unicode code point。
    # 这里在不依赖测试工具函数的前提下重建它的反向映射。
    bytes_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    character_values = bytes_values[:]
    offset = 0
    for byte_value in range(256):
        if byte_value not in bytes_values:
            bytes_values.append(byte_value)
            character_values.append(256 + offset)
            offset += 1

    return {chr(character_value): byte_value for byte_value, character_value in zip(bytes_values, character_values)}
