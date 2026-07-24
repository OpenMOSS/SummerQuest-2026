from __future__ import annotations

import json
from pathlib import Path


def save_vocab(vocab: dict[int, bytes], output_path: str | Path) -> None:
    """将任意 bytes token 安全地序列化为十六进制字符串。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(token_id): token_bytes.hex() for token_id, token_bytes in sorted(vocab.items())}
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def save_merges(merges: list[tuple[bytes, bytes]], output_path: str | Path) -> None:
    """每行保存两个 hex token；空格分隔，适合 from_files 逐行 split。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for left, right in merges:
            file.write(f"{left.hex()} {right.hex()}\n")
