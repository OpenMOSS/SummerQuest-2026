from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# 把脚本的父目录（assignment1-basics/）加入 import 搜索路径，确保能 import cs336_basics
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import train_bpe


def parse_args() -> argparse.Namespace:
    # 命令行参数：--input 语料 / --vocab-size 目标词表 / --special-tokens 特殊token / --out-dir 输出目录
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text corpus.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the input text corpus (e.g. tinystories_train.txt)")
    parser.add_argument("--vocab-size", required=True, type=int,
                        help="Target vocab size (includes special tokens). 10000 for TS, 32000 for OWT.")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"],
                        help="Space-separated special tokens. Default: '<|endoftext|>'.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Output dir for tokenizer.pkl / vocab.json / merges.txt / config.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ---- 基础参数校验 ----
    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1
    if args.vocab_size <= 256 + len(args.special_tokens):
        print(f"ERROR: vocab_size ({args.vocab_size}) must be > 256 + num_special_tokens", file=sys.stderr)
        return 1

    # ---- 准备输出目录 ----
    args.out_dir.mkdir(parents=True, exist_ok=True)  # parents=True 支持递归建目录

    print(f"[train_tokenizer] input={args.input}", flush=True)
    print(f"[train_tokenizer] vocab_size={args.vocab_size}, special_tokens={args.special_tokens}", flush=True)
    print(f"[train_tokenizer] out_dir={args.out_dir}", flush=True)

    # ---- 核心：调用你已经实现的 train_bpe 纯函数 ----
    t0 = time.time()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_tokens)
    elapsed = time.time() - t0
    print(f"[train_tokenizer] train_bpe finished in {elapsed:.1f}s. vocab={len(vocab)}, merges={len(merges)}", flush=True)

    # ---- 产物 1/4：二进制 tokenizer.pkl（encode 和 eval 时 load 回来用这个，保留 bytes 类型） ----
    pkl_path = args.out_dir / "tokenizer.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": args.special_tokens}, f)
    print(f"[train_tokenizer] 1/4 pickle -> {pkl_path}", flush=True)

    # ---- 产物 2/4：vocab.json（人类可读，bytes 转成 list[int] 方便 JSON 存） ----
    vocab_for_json = {str(k): list(v) for k, v in vocab.items()}
    json_path = args.out_dir / "vocab.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_for_json, f, ensure_ascii=False, indent=2)
    print(f"[train_tokenizer] 2/4 vocab.json -> {json_path}", flush=True)

    # ---- 产物 3/4：merges.txt（每一行是「左bytes十六进制 右bytes十六进制」，顺序即 merge 优先级） ----
    merges_path = args.out_dir / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(left.hex() + " " + right.hex() + "\n")
    print(f"[train_tokenizer] 3/4 merges.txt (hex) -> {merges_path}", flush=True)

    # ---- 产物 4/4：config.json（超参元数据，README 表格里的数据可以从这里抄） ----
    config = {
        "input": str(args.input),
        "vocab_size_target": args.vocab_size,
        "vocab_size_actual": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_tokens,
        "train_time_sec": round(elapsed, 2),
    }
    config_path = args.out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[train_tokenizer] 4/4 config.json -> {config_path}", flush=True)

    # ---- 语料太小没到目标词表时的告警 ----
    if len(vocab) < args.vocab_size:
        print(
            f"WARNING: actual vocab {len(vocab)} < target {args.vocab_size}. "
            f"Corpus too small; use smaller --vocab-size or larger corpus.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
