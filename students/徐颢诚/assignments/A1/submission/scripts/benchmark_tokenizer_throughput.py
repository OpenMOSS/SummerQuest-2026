"""测量 tokenizer 的单进程编码吞吐量，并估算处理 Pile 所需时间。"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


DEFAULT_TOKENIZER_PATH = Path("artifacts/owt_bpe_32k.pkl")
DEFAULT_INPUT_PATH = Path("data/owt_valid.txt")
PILE_SIZE_BYTES = 825e9


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """加载训练脚本序列化的 BPE tokenizer。"""
    with tokenizer_path.open("rb") as tokenizer_file:
        serialized = pickle.load(tokenizer_file)
    return Tokenizer(
        serialized["vocab"],
        serialized["merges"],
        serialized.get("special_tokens", ["<|endoftext|>"]),
    )


def main() -> None:
    """对给定文本文件进行一次流式编码并打印吞吐量。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    arguments = parser.parse_args()

    if not arguments.tokenizer.is_file():
        raise FileNotFoundError(f"找不到 tokenizer 文件：{arguments.tokenizer}")
    if not arguments.input.is_file():
        raise FileNotFoundError(f"找不到输入文件：{arguments.input}")

    tokenizer = load_tokenizer(arguments.tokenizer)
    byte_count = 0
    token_count = 0
    start = time.perf_counter()
    with arguments.input.open(encoding="utf-8") as input_file:
        for line in input_file:
            byte_count += len(line.encode("utf-8"))
            token_count += sum(1 for _ in tokenizer.iter_encode(line))
    elapsed_seconds = time.perf_counter() - start
    bytes_per_second = byte_count / elapsed_seconds

    print(f"输入文件：{arguments.input}")
    print(f"输入大小：{byte_count:,} bytes")
    print(f"token 数：{token_count:,}")
    print(f"耗时：{elapsed_seconds:.2f} 秒")
    print(f"吞吐量：{bytes_per_second / 1e6:.2f} MB/s")
    print(f"吞吐量：{bytes_per_second / 2**20:.2f} MiB/s")
    print(f"按此单进程吞吐量处理 825GB Pile 的预计时间：{PILE_SIZE_BYTES / bytes_per_second / 3600:.2f} 小时")


if __name__ == "__main__":
    main()
