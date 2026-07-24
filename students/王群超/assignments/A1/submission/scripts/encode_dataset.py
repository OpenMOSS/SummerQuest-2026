from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# 加入父目录 import 搜索路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    # 参数：--tokenizer (pkl) / --input (.txt) / --output (.npy) / --progress-every
    parser = argparse.ArgumentParser(
        description="Stream-encode a text file into a numpy .npy token-id file."
    )
    parser.add_argument("--tokenizer", required=True, type=Path,
                        help="Path to tokenizer.pkl (vocab+merges+special_tokens).")
    parser.add_argument("--input", required=True, type=Path, help="Input .txt corpus.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npy path (overwritten).")
    parser.add_argument("--progress-every", type=int, default=10_000_000,
                        help="Print progress every N tokens. Default: 10M.")
    return parser.parse_args()


def load_tokenizer(pkl_path: Path) -> BPETokenizer:
    # 兼容两种 pickle 格式：① train_tokenizer.py 写的 dict；② 旧版 tuple(vocab, merges)
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple):
        vocab, merges = obj
        special_tokens = None
    elif isinstance(obj, dict):
        vocab = obj["vocab"]; merges = obj["merges"]; special_tokens = obj.get("special_tokens")
    else:
        raise ValueError(f"Unsupported pickle format in {pkl_path}: got {type(obj)}")
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def main() -> int:
    args = parse_args()

    # ---- 参数校验 ----
    if not args.tokenizer.is_file():
        print(f"ERROR: tokenizer not found: {args.tokenizer}", file=sys.stderr); return 1
    if not args.input.is_file():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr); return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- 加载 tokenizer，根据 vocab 选数据类型：≤65535 用 uint16，否则 uint32 ----
    print(f"[encode_dataset] Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer.id_to_bytes)
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    print(f"[encode_dataset] vocab={vocab_size}, dtype={dtype.__name__}")

    input_size_bytes = args.input.stat().st_size
    print(f"[encode_dataset] input={args.input} ({input_size_bytes/1e9:.2f} GB) → output={args.output}")

    # ---- 可选 tqdm 进度条：装了就用，没装就原样返回（不报错） ----
    try:
        from tqdm import tqdm
        progress_wrapper = lambda it: tqdm(it, desc="encoding", unit="tok")
    except ImportError:
        progress_wrapper = lambda it: it

    # ---- ⭐ 核心：流式编码。encode_iterable 每次 yield 一个 id，不会把整个 txt 载入内存 ----
    ids: list[int] = []
    t0 = time.time()
    last_report_tokens, last_report_time = 0, t0
    with open(args.input, "r", encoding="utf-8") as f:
        for _id in progress_wrapper(tokenizer.encode_iterable(f)):
            ids.append(_id)
            # 周期性打印进度：速度、已用时间、估算剩余（乘 1.05 留点余量）
            if args.progress_every > 0 and len(ids) - last_report_tokens >= args.progress_every:
                now = time.time()
                dt, dtok = now - last_report_time, len(ids) - last_report_tokens
                tok_per_sec = dtok / dt if dt > 0 else 0.0
                elapsed = now - t0
                est_total = (elapsed / len(ids)) * len(ids) * 1.05 if len(ids) > 0 else 0
                print(f"  {len(ids)/1e6:.2f}M tok | {tok_per_sec/1e3:.1f}k/s | "
                      f"{elapsed:.0f}s done | {max(0, est_total-elapsed):.0f}s left")
                last_report_tokens, last_report_time = len(ids), now

    # ---- 汇总统计 ----
    total_time = time.time() - t0
    n_tokens = len(ids)
    print(f"[encode_dataset] Done: {n_tokens:,} tokens in {total_time:.1f}s "
          f"({n_tokens/total_time:,.0f} tok/s)" if total_time > 0 else "")
    if n_tokens > 0:
        print(f"[encode_dataset] Compression: {input_size_bytes}/{n_tokens} = "
              f"{input_size_bytes/n_tokens:.3f} bytes/token")

    # ---- 写 numpy 文件；先转数组再 del ids 防止内存翻倍时 OOM ----
    arr = np.array(ids, dtype=dtype)
    del ids
    print(f"[encode_dataset] Array: shape={arr.shape}, {arr.nbytes/1e9:.3f} GB on disk")
    np.save(args.output, arr)
    print(f"[encode_dataset] Saved npy -> {args.output}")

    # ---- 写 companion meta.json（README 的数据可以从这里直接抄） ----
    import json
    meta = {
        "tokenizer": str(args.tokenizer), "input": str(args.input),
        "input_bytes": input_size_bytes, "output": str(args.output),
        "vocab_size": vocab_size, "dtype": dtype.__name__, "num_tokens": int(n_tokens),
        "array_nbytes": int(arr.nbytes),
        "compression_ratio_bytes_per_token": round(input_size_bytes/n_tokens, 4) if n_tokens else None,
        "encode_time_sec": round(total_time, 2),
        "tokens_per_sec": round(n_tokens/total_time, 2) if total_time else None,
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[encode_dataset] Saved meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
