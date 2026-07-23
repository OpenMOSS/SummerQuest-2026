from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

# 加入父目录 import 搜索路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-domain tokenizer comparison"
    )
    parser.add_argument("--tokenizer", action="append", required=True,
                        metavar="NAME=PATH",
                        help="Repeatable. e.g. --tokenizer 'TS_10K=a.pkl' --tokenizer 'OWT_32K=b.pkl'")
    parser.add_argument("--ts-corpus", type=Path, required=True,
                        help="TinyStories corpus path")
    parser.add_argument("--owt-corpus", type=Path, required=True,
                        help="OpenWebText corpus path")
    parser.add_argument("--n-docs", type=int, default=10,
                        help="Number of documents to sample per corpus (total = 2 * n_docs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Save full results as JSON")
    return parser.parse_args()


def load_tokenizer(pkl_path: Path) -> BPETokenizer:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple):
        vocab, merges = obj
        special_tokens = None
    elif isinstance(obj, dict):
        vocab = obj["vocab"]
        merges = obj["merges"]
        special_tokens = obj.get("special_tokens")
    else:
        raise ValueError(f"Unsupported pickle format in {pkl_path}: {type(obj)}")
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def sample_documents(filepath: Path, n_docs: int, seed: int) -> list[str]:

    random.seed(seed)
    delimiter = "<|endoftext|>"

    # 第一遍：统计文档数
    print(f"[sample] counting documents in {filepath}...", flush=True)
    doc_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(64 * 1024 * 1024)
            if not chunk:
                break
            doc_count += chunk.count(delimiter)
    print(f"[sample]   total ~{doc_count} documents", flush=True)

    # 随机选文档索引
    selected = sorted(random.sample(range(doc_count), min(n_docs, doc_count)))
    selected_set = set(selected)
    print(f"[sample]   selected indices: {selected}", flush=True)

    # 第二遍：只存被选中的文档
    selected_docs: list[str] = []
    current_doc: list[str] = []
    doc_idx = 0
    with open(filepath, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(64 * 1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            while delimiter in buffer:
                split_idx = buffer.index(delimiter)
                current_doc.append(buffer[:split_idx])
                buffer = buffer[split_idx + len(delimiter):]

                if doc_idx in selected_set:
                    doc = "".join(current_doc).strip()
                    if doc:
                        selected_docs.append(doc)
                current_doc = []
                doc_idx += 1
    # 最后一段
    if buffer.strip():
        current_doc.append(buffer)
        if doc_idx in selected_set:
            doc = "".join(current_doc).strip()
            if doc:
                selected_docs.append(doc)

    print(f"[sample]   collected {len(selected_docs)} documents, "
          f"total {sum(len(d) for d in selected_docs):,} chars", flush=True)
    return selected_docs


def main() -> int:
    args = parse_args()

    # ---- 加载 tokenizer ----
    tokenizers: list[tuple[str, BPETokenizer]] = []
    for spec in args.tokenizer:
        name, path_str = spec.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip())
        if not path.is_file():
            print(f"ERROR: tokenizer not found: {path}", file=sys.stderr)
            return 1
        tok = load_tokenizer(path)
        tokenizers.append((name, tok))
        print(f"[compare] loaded tokenizer: {name} (vocab={len(tok.id_to_bytes)})", flush=True)

    # ---- 采样文档（TS + OWT 混成一个统一数据集） ----
    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Sampling from TinyStories: {args.ts_corpus}")
    print(f"{'='*60}")
    ts_docs = sample_documents(args.ts_corpus, args.n_docs, args.seed)

    print(f"\n{'='*60}")
    print(f"Sampling from OpenWebText: {args.owt_corpus}")
    print(f"{'='*60}")
    owt_docs = sample_documents(args.owt_corpus, args.n_docs, args.seed + 1)

    # 混合数据集
    all_docs = ts_docs + owt_docs
    random.shuffle(all_docs)
    combined_text = "\n".join(all_docs)
    combined_bytes = len(combined_text.encode("utf-8"))
    print(f"\n[compare] unified dataset: {len(ts_docs)} TS + {len(owt_docs)} OWT docs, "
          f"{combined_bytes:,} bytes", flush=True)

    # ---- 逐个 tokenizer 编码 ----
    rows = []
    for tok_name, tok_obj in tokenizers:
        print(f"\n=== Encoding with {tok_name} ===")

        # 最长 token
        longest_len = max(len(bs) for bs in tok_obj.id_to_bytes.values())
        longest_id = max(
            tok_obj.id_to_bytes.keys(),
            key=lambda i: len(tok_obj.id_to_bytes[i])
        )
        longest_bytes = tok_obj.id_to_bytes[longest_id]
        longest_preview = longest_bytes.decode("utf-8", errors="replace")
        print(f"  longest_token: len={longest_len}B, id={longest_id}, "
              f"preview={longest_preview[:80]!r}")

        # 编码
        t0 = time.perf_counter()
        ids = tok_obj.encode(combined_text)
        elapsed = time.perf_counter() - t0

        n_tokens = len(ids)
        comp_ratio = combined_bytes / n_tokens if n_tokens else float("nan")
        tps = n_tokens / elapsed if elapsed else 0.0

        print(f"  tokens={n_tokens:,}  time={elapsed:.2f}s")
        print(f"  compression = {comp_ratio:.3f} bytes/token")
        print(f"  throughput  = {tps:,.0f} tok/s")

        rows.append({
            "tokenizer": tok_name,
            "vocab_size": len(tok_obj.id_to_bytes),
            "total_bytes": combined_bytes,
            "total_tokens": n_tokens,
            "compression_ratio_bytes_per_token": round(comp_ratio, 4),
            "encode_time_sec": round(elapsed, 4),
            "tokens_per_sec": round(tps, 2),
            "longest_token_bytes": longest_len,
            "longest_token_preview": longest_preview,
            "longest_token_id": longest_id,
        })

    # ---- Markdown 表格 ----
    print(f"\n{'='*80}")
    print("TOKENIZER COMPARISON ON MIXED CORPUS")
    print(f"({len(ts_docs)} TS docs + {len(owt_docs)} OWT docs, {combined_bytes:,} bytes)")
    print(f"{'='*80}")
    print()

    header = (f"| {'Metric':<28} | {'TS_10K':>22} | {'OWT_32K':>22} |")
    sep = "|" + "|".join("-" * (len(c) + 2) for c in header.strip("|").split("|")) + "|"
    print(header)
    print(sep)

    # 需要两个 tokenizer 都存在
    ts_row = next((r for r in rows if "TS" in r["tokenizer"]), None)
    owt_row = next((r for r in rows if "OWT" in r["tokenizer"]), None)

    def print_metric(label: str, ts_val, owt_val, fmt: str = ","):
        if isinstance(ts_val, float):
            tsv = f"{ts_val:{fmt}}" if "f" not in fmt else f"{ts_val:{fmt}}"
            owv = f"{owt_val:{fmt}}" if "f" not in fmt else f"{owt_val:{fmt}}"
        else:
            tsv = f"{ts_val:{fmt}}"
            owv = f"{owt_val:{fmt}}"
        print(f"| {label:<28} | {tsv:>22} | {owv:>22} |")

    if ts_row and owt_row:
        print_metric("Vocab size", ts_row["vocab_size"], owt_row["vocab_size"])
        print_metric("Total tokens", ts_row["total_tokens"], owt_row["total_tokens"])
        print_metric("Compression (B/tok)", ts_row["compression_ratio_bytes_per_token"],
                     owt_row["compression_ratio_bytes_per_token"], ".3f")
        print_metric("Throughput (tok/s)", ts_row["tokens_per_sec"],
                     owt_row["tokens_per_sec"], ".0f")
        print_metric("Longest token (bytes)", ts_row["longest_token_bytes"],
                     owt_row["longest_token_bytes"])
        print()
        print(f"Longest token previews:")
        print(f"  TS_10K: {ts_row['longest_token_preview'][:120]!r}")
        print(f"  OWT_32K: {owt_row['longest_token_preview'][:120]!r}")

    # ---- 保存 JSON ----
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({
                "ts_corpus": str(args.ts_corpus),
                "owt_corpus": str(args.owt_corpus),
                "n_docs_per_corpus": args.n_docs,
                "seed": args.seed,
                "combined_bytes": combined_bytes,
                "results": rows,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[compare] saved to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
