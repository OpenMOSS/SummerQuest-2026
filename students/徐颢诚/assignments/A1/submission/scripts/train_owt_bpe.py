"""训练并保存 OpenWebText 的 32K 字节级 BPE 分词器。"""

from __future__ import annotations

import pickle
import time
import os
from pathlib import Path

from cs336_basics.tokenizer import train_bpe_tokenizer


INPUT_PATH = Path("data/owt_train.txt")
OUTPUT_PATH = Path("artifacts/owt_bpe_32k.pkl")
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_PROCESSES = min(32, os.cpu_count() or 1)


def main() -> None:
    """训练 BPE、写入结果，并输出 writeup 所需的基本统计。"""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"找不到 OpenWebText 训练文件：{INPUT_PATH}")

    start = time.perf_counter()
    vocab, merges = train_bpe_tokenizer(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        progress=True,
        num_processes=NUM_PROCESSES,
    )
    elapsed_seconds = time.perf_counter() - start

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as output_file:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": SPECIAL_TOKENS,
                "vocab_size": VOCAB_SIZE,
            },
            output_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    longest_length = max(map(len, vocab.values()))
    longest_tokens = [token for token in vocab.values() if len(token) == longest_length]

    print(f"\n训练完成，耗时：{elapsed_seconds / 60:.2f} 分钟")
    print(f"词表大小：{len(vocab):,}")
    print(f"merge 数量：{len(merges):,}")
    print(f"结果已保存到：{OUTPUT_PATH}")
    print(f"最长 token 字节长度：{longest_length}")
    print("最长 token（bytes）：")
    for token in longest_tokens:
        print(f"  {token!r}")
    print("最长 token（UTF-8 尝试解码）：")
    for token in longest_tokens:
        print(f"  {token.decode('utf-8', errors='replace')!r}")


if __name__ == "__main__":
    main()
