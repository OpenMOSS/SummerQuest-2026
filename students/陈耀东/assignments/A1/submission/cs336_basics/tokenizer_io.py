"""BPE tokenizer 词表与 merge 规则的可读文件格式。"""

from __future__ import annotations

import json
import os
from pathlib import Path


def save_tokenizer_files(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """把 tokenizer 保存为两个 JSON 文件。

    bytes 不能直接写入 JSON，因此使用十六进制字符串保存。
    这种格式可读、可比较，也能无损恢复任意 byte 序列。
    """
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    merges_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_vocab = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in sorted(vocab.items())
    }
    serialized_merges = [
        [left.hex(), right.hex()]
        for left, right in merges
    ]

    vocab_path.write_text(
        json.dumps(serialized_vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    merges_path.write_text(
        json.dumps(serialized_merges, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tokenizer_files(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从 ``save_tokenizer_files`` 生成的 JSON 文件恢复 tokenizer。"""
    serialized_vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    serialized_merges = json.loads(Path(merges_path).read_text(encoding="utf-8"))

    vocab = {
        int(token_id): bytes.fromhex(token_hex)
        for token_id, token_hex in serialized_vocab.items()
    }
    merges = [
        (bytes.fromhex(left_hex), bytes.fromhex(right_hex))
        for left_hex, right_hex in serialized_merges
    ]
    return vocab, merges
