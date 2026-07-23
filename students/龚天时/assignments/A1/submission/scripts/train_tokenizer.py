import argparse
import json
import os
import pickle
import resource
import sys
import time

from cs336_basics.tokenizer import train_bpe


def get_peak_memory_gb():
    """峰值内存。注意:macOS 返回 bytes,Linux 返回 KB。"""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return peak_bytes / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="语料文件路径")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--name", required=True, help="输出前缀,如 tinystories")
    parser.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ---- 训练----
    t0 = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_tokens)
    train_time = time.perf_counter() - t0

    peak_gb = get_peak_memory_gb()

    # ---- 存盘----
    with open(f"artifacts/{args.name}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"artifacts/{args.name}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # ---- 报告要的指标 ----
    longest = max(vocab.values(), key=len)

    log = {
        "dataset": args.name,
        "vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_tokens,
        "train_time_sec": round(train_time, 2),
        "peak_memory_gb": round(peak_gb, 3),
        "longest_token_len": len(longest),
        "longest_token_repr": repr(longest),
        "longest_token_decoded": longest.decode("utf-8", errors="replace"),
    }

    with open(f"logs/tokenizer_{args.name}.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":      
    main()