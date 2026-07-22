import argparse
import json
import os
import pickle
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="要编码的文本文件")
    parser.add_argument("--vocab", required=True, help="vocab pkl 路径")
    parser.add_argument("--merges", required=True, help="merges pkl 路径")
    parser.add_argument("--output", required=True, help="输出 .npy 路径")
    parser.add_argument("--name", required=True, help="日志里的标识名")
    args = parser.parse_args()


    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    t0 = time.perf_counter()
    chunks = []            # 存若干个 numpy 小数组
    buffer = []            # 临时攒 token id
    total_tokens = 0

    with open(args.input, encoding="utf-8") as f:
        for tok_id in tokenizer.encode_iterable(f):
            buffer.append(tok_id)
            if len(buffer) >= 1_000_000:          # 每攒够 100 万个就转成 uint16
                chunks.append(np.array(buffer, dtype=np.uint16))
                total_tokens += len(buffer)
                buffer = []
    if buffer:                                     # 剩下不足 100 万的
        chunks.append(np.array(buffer, dtype=np.uint16))
        total_tokens += len(buffer)

    arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    elapsed = time.perf_counter() - t0

    if len(arr) > 0:
        assert arr.max() < 65536, "token id 超出 uint16 范围!"
    np.save(args.output, arr)

    total_bytes = os.path.getsize(args.input)
    log = {
        "name": args.name,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "compression_ratio": round(total_bytes / total_tokens, 4),
        "throughput_bytes_per_sec": round(total_bytes / elapsed),
        "encode_time_sec": round(elapsed, 2),
        "output_file": args.output,
    }

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/encode_{args.name}.json", "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()