import os
from pathlib import Path
import pickle
import time

from cs336_basics.tokenizer import train_bpe_tokenizer

NUM_PROCESSES = min(32, os.cpu_count() or 1)

def main() -> None:
    start = time.perf_counter()
    vocab, merges = train_bpe_tokenizer(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        progress=True,
        num_processes=NUM_PROCESSES,
    )

    elapsed = time.perf_counter() - start

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "tinystories_bpe_10k.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    longest_tokens = sorted(vocab.values(), key=len, reverse=True)
    max_length = len(longest_tokens[0])
    longest = [token for token in longest_tokens if len(token) == max_length]

    print(f"训练时间：{elapsed:.2f} 秒")
    print(f"最长 token 字节数：{max_length}")
    print(f"最长 token：{longest}")


if __name__ == "__main__":
    main()
