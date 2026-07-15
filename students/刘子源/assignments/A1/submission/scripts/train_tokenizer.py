# tokenizer 训练入口，参数不少，实际干活的还是 train_bpe。

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from cs336_basics.tokenizer import Tokenizer, train_bpe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="预分词进程数；默认根据文件大小和 SLURM_CPUS_PER_TASK 自动选择",
    )
    parser.add_argument(
        "--chunks-per-worker",
        type=int,
        default=4,
        help="每个 worker 对应的安全文件块数量，用于改善负载均衡",
    )
    parser.add_argument("--quiet", action="store_true", help="关闭 BPE 阶段进度输出")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file does not exist or is not a regular file: {args.input}")
    if args.vocab_size <= 0:
        parser.error("--vocab-size must be positive")
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.chunks_per_worker < 1:
        parser.error("--chunks-per-worker must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        print(
            f"[BPE] input={args.input} size={args.input.stat().st_size / 2**30:.2f} GiB "
            f"vocab_size={args.vocab_size} workers={args.num_workers or 'auto'}",
            file=sys.stderr,
            flush=True,
        )
    started = time.perf_counter()
    vocab, merges = train_bpe(
        args.input,
        args.vocab_size,
        args.special_token,
        num_workers=args.num_workers,
        chunks_per_worker=args.chunks_per_worker,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - started
    tokenizer = Tokenizer(vocab, merges, args.special_token)
    tokenizer.save(args.output_dir / "vocab.json", args.output_dir / "merges.json")
    # 不搞 key=lambda 了，最长的挨个比，谁都看得懂。
    longest = b""
    for token in vocab.values():
        token_bytes = bytes(token)
        if len(token_bytes) > len(longest):
            longest = token_bytes
    metadata = {
        "input": str(args.input),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_token,
        "requested_num_workers": args.num_workers if args.num_workers is not None else "auto",
        "chunks_per_worker": args.chunks_per_worker,
        "training_time_sec": elapsed,
        "longest_token_bytes_hex": longest.hex(),
        "longest_token_decoded": longest.decode("utf-8", errors="replace"),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Windows 多进程就认这套，别删，服务器没事不代表本地没事。
    multiprocessing.freeze_support()
    main()
