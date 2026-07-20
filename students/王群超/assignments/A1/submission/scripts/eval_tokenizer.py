from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# 加入父目录 import 搜索路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    # --corpus 共同评测语料；--tokenizer NAME=PATH 可重复传多次；--stream 用流式；--out-json 存汇总
    parser = argparse.ArgumentParser(
        description="Compare compression ratio, throughput and longest token across tokenizers."
    )
    parser.add_argument("--corpus", required=True, type=Path,
                        help="Shared eval corpus (e.g. tinystories_dev.txt).")
    parser.add_argument("--tokenizer", action="append", required=True, metavar="NAME=PATH",
                        help="Repeatable, e.g. --tokenizer 'TS_10K=a.pkl' --tokenizer 'OWT_32K=b.pkl'.")
    parser.add_argument("--stream", action="store_true",
                        help="Use encode_iterable instead of in-memory encode().")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Optional path to save full metrics as JSON.")
    return parser.parse_args()


def _parse_tokenizer_spec(spec: str) -> tuple[str, Path]:
    # split("=", 1) 第 2 个参数 = 只切 1 刀，避免路径里含等号时被切碎
    if "=" not in spec:
        raise ValueError(f"--tokenizer must be 'NAME=PATH', got {spec!r}")
    name, path_str = spec.split("=", 1)
    return name.strip(), Path(path_str)


def load_tokenizer(pkl_path: Path) -> BPETokenizer:
    # 兼容 dict 格式（本仓库写的）和 tuple 格式（外部格式）
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple):
        vocab, merges = obj; special_tokens = None
    elif isinstance(obj, dict):
        vocab = obj["vocab"]; merges = obj["merges"]; special_tokens = obj.get("special_tokens")
    else:
        raise ValueError(f"Unsupported pickle format in {pkl_path}: {type(obj)}")
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def longest_token_bytes(tokenizer: BPETokenizer) -> tuple[int, int, bytes]:
    # 遍历 vocab 找最长（bytes 长度最大）的 token。写 README 时可以拿这个举例子。
    best_id, best_len, best_bytes = -1, -1, b""
    for _id, bs in tokenizer.id_to_bytes.items():
        if len(bs) > best_len:
            best_len, best_id, best_bytes = len(bs), _id, bs
    return best_id, best_len, best_bytes


def main() -> int:
    args = parse_args()

    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr); return 1

    # ---- 准备：拿语料字节数 + 解析所有 --tokenizer 对 ----
    corpus_bytes = os.path.getsize(args.corpus)
    print(f"[eval_tokenizer] corpus={args.corpus} ({corpus_bytes/1e6:.2f} MB)")
    print(f"[eval_tokenizer] mode={'stream' if args.stream else 'in-memory'}")
    specs: list[tuple[str, Path]] = []
    for spec in args.tokenizer:
        name, path = _parse_tokenizer_spec(spec)
        if not path.is_file():
            print(f"ERROR: tokenizer not found: {path} (name={name})", file=sys.stderr); return 1
        specs.append((name, path))

    # ---- 非流式模式：把语料整个读进内存（dev 集一般可以） ----
    corpus_text = None
    if not args.stream:
        print("[eval_tokenizer] Reading corpus into RAM for encode()...")
        with open(args.corpus, "r", encoding="utf-8") as f:
            corpus_text = f.read()
        print(f"[eval_tokenizer]   {len(corpus_text)} chars")

    # ---- ⭐ 核心：逐个 tokenizer 跑 4 个指标 ----
    rows = []
    for name, path in specs:
        print(f"\n=== {name} ({path}) ===")
        tok = load_tokenizer(path)

        # 指标 1：vocab 大小
        # 指标 2：最长 token（bytes 长度 + 预览）
        longest_id, longest_len, longest_bs = longest_token_bytes(tok)
        longest_preview = longest_bs.decode("utf-8", errors="replace")
        print(f"  vocab={len(tok.id_to_bytes)}")
        print(f"  longest_token: len={longest_len}B, id={longest_id}, preview={longest_preview!r}")

        # 指标 3&4：编码一遍，同时量 throughput（时间测不准的话自己多跑几次取平均）
        t0 = time.perf_counter()  # perf_counter 比 time.time 更准
        n_tokens = 0
        if args.stream:
            with open(args.corpus, "r", encoding="utf-8") as f:
                for _id in tok.encode_iterable(f):
                    n_tokens += 1
        else:
            ids = tok.encode(corpus_text)  # type: ignore[arg-type]
            n_tokens = len(ids)
        elapsed = time.perf_counter() - t0

        comp_ratio = corpus_bytes / n_tokens if n_tokens else float("nan")
        tps = n_tokens / elapsed if elapsed else 0.0
        bps = corpus_bytes / elapsed / 1e6 if elapsed else 0.0
        print(f"  tokens={n_tokens:,}  time={elapsed:.2f}s")
        print(f"  compression = {comp_ratio:.3f} bytes/token")
        print(f"  throughput  = {tps:,.0f} tok/s  ({bps:.2f} MB/s)")

        rows.append({
            "name": name, "tokenizer_path": str(path),
            "vocab_size": len(tok.id_to_bytes),
            "longest_token_bytes": longest_len, "longest_token_preview": longest_preview,
            "corpus_bytes": corpus_bytes, "num_tokens": n_tokens,
            "compression_ratio_bytes_per_token": round(comp_ratio, 4),
            "encode_elapsed_sec": round(elapsed, 4),
            "tokens_per_sec": round(tps, 2), "mb_per_sec": round(bps, 4),
            "stream_mode": args.stream,
        })

    # ---- 打印对齐的 Markdown 风格表格 ----
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (copy this into README)")
    print("=" * 100)
    header = f"| {'Name':<20} | {'Vocab':>7} | {'Tokens':>12} | {'Comp(B/tok)':>12} | {'Tok/s':>12} | {'Longest(B)':>10} |"
    sep = "|" + "|".join("-" * (len(c) + 2) for c in header.strip("|").split("|")) + "|"
    print(header); print(sep)
    for r in rows:
        print(f"| {r['name']:<20} | {r['vocab_size']:>7,} | {r['num_tokens']:>12,} | "
              f"{r['compression_ratio_bytes_per_token']:>12.3f} | {r['tokens_per_sec']:>11,.0f} | "
              f"{r['longest_token_bytes']:>10} |")
    print("=" * 100)

    # ---- 可选：写 JSON 汇总（README 要数字就从这里拿） ----
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"corpus": str(args.corpus), "corpus_bytes": corpus_bytes,
                       "stream_mode": args.stream, "results": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\n[eval_tokenizer] JSON -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
