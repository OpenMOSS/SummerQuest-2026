import os
import time
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import psutil

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def get_memory_usage_mb():
    """获取当前进程内存占用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def write_log(log_dir, log_entry):
    """将一条日志追加写入 log_dir 下的 JSONL 文件"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "tokenizer_tinystories.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def main(args):
    data_dir = Path("data/TinyStories")
    train_text_path = data_dir / args.train_file
    val_text_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    output_dir = Path("data/tinystories_tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # 记录训练开始
    start_time = time.time()
    mem_before = get_memory_usage_mb()
    print("Training BPE tokenizer...")
    write_log(log_dir, {
        "event": "train_start",
        "timestamp": time.time(),
        "input_path": str(train_text_path),
        "vocab_size": vocab_size,
    })

    # ---------- 训练 BPE ----------
    vocab, merges = train_bpe(
        input_path=str(train_text_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    elapsed = time.time() - start_time
    mem_after = get_memory_usage_mb()
    print(f"Training took {elapsed:.2f} seconds.")
    print(f"Memory during training: {mem_after - mem_before:.2f} MB (peak {mem_after:.2f} MB)")
    write_log(log_dir, {
        "event": "train_complete",
        "train_time_sec": elapsed,
        "memory_peak_mb": max(mem_after, mem_before),
    })

    # 保存 vocab 和 merges
    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # 最长 token
    longest_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_bytes = vocab[longest_id]
    print(f"Longest token ID: {longest_id}, bytes: {longest_bytes!r}, length: {len(longest_bytes)} bytes")
    write_log(log_dir, {
        "event": "longest_token",
        "token_id": longest_id,
        "token_bytes_length": len(longest_bytes),
        "token_bytes_utf8": longest_bytes.decode("utf-8", errors="replace"),
    })

    # ---------- 构建 tokenizer ----------
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # ---------- 编码验证集 ----------
    print("Encoding validation set...")
    val_raw_size = val_text_path.stat().st_size
    start_val = time.time()
    val_tokens = encode_file(tokenizer, val_text_path)
    val_elapsed = time.time() - start_val
    val_out = output_dir / "val_tokens.npy"
    np.save(val_out, val_tokens.astype(np.uint16))

    # ---------- 编码训练集 ----------
    print("Encoding training set...")
    train_raw_size = train_text_path.stat().st_size
    start_train = time.time()
    train_tokens = encode_file(tokenizer, train_text_path)
    train_elapsed = time.time() - start_train
    train_out = output_dir / "train_tokens.npy"
    np.save(train_out, train_tokens.astype(np.uint16))

    # 计算指标
    val_num_tokens = len(val_tokens)
    train_num_tokens = len(train_tokens)

    val_compression = val_raw_size / val_num_tokens if val_num_tokens > 0 else float('inf')
    train_compression = train_raw_size / train_num_tokens if train_num_tokens > 0 else float('inf')

    val_throughput = val_num_tokens / val_elapsed if val_elapsed > 0 else 0
    train_throughput = train_num_tokens / train_elapsed if train_elapsed > 0 else 0

    metrics = {
        "event": "metrics",
        "train_time_sec": elapsed,
        "memory_peak_mb": max(mem_after, mem_before),
        "longest_token_bytes_length": len(longest_bytes),
        "val_raw_bytes": val_raw_size,
        "val_num_tokens": val_num_tokens,
        "val_compression_ratio": val_compression,
        "val_throughput_tokens_per_sec": val_throughput,
        "train_raw_bytes": train_raw_size,
        "train_num_tokens": train_num_tokens,
        "train_compression_ratio": train_compression,
        "train_throughput_tokens_per_sec": train_throughput,
    }

    print("\n===== Tokenizer Metrics =====")
    for key, value in metrics.items():
        if key != "event":
            print(f"{key}: {value}")
    write_log(log_dir, metrics)

    print(f"Logs saved to {log_dir / 'tokenizer_tinystories.jsonl'}")


def encode_file(tokenizer, file_path, chunk_size=1024*1024):
    """流式编码文件"""
    ids_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        def text_chunks():
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        for token_id in tokenizer.encode_iterable(text_chunks()):
            ids_list.append(token_id)
    return np.array(ids_list, dtype=np.uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="TinyStoriesV2-GPT4-train.txt")

    # 让默认日志目录指向项目根目录下的 logs
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_log_dir = project_root / "log"

    parser.add_argument("--log_dir", type=str, default=str(default_log_dir))
    args = parser.parse_args()
    main(args)