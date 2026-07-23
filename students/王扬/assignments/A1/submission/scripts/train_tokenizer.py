from __future__ import annotations

import argparse
import time
from pathlib import Path

from cs336_basics.tokenizer import BPETokenizer, train_bpe

from scripts.tokenizer_io import save_tokenizer
from scripts.utils import Timer, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    input_path = cfg["input_path"]
    vocab_size = int(cfg["vocab_size"])
    special_tokens = list(cfg.get("special_tokens", []))
    progress = bool(cfg.get("progress", True))
    num_workers = int(cfg.get("num_workers", 1))

    t = Timer.start_now()
    if progress:
        print(
            f"[train_tokenizer] input={input_path} vocab_size={vocab_size} special_tokens={special_tokens} num_workers={num_workers}",
            flush=True,
        )
        print("[train_tokenizer] stage=pretokenize_then_merge", flush=True)
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        progress=progress,
        num_workers=num_workers,
    )
    if progress:
        print("[train_tokenizer] stage=save_tokenizer", flush=True)
    save_tokenizer(args.output_dir, vocab=vocab, merges=merges, special_tokens=special_tokens)

    tok = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    sample_path = cfg.get("stats_sample_path", input_path)
    sample_bytes = int(cfg.get("stats_sample_bytes", 5_000_000))
    text = Path(sample_path).read_text(encoding="utf-8")[:sample_bytes]

    token_ids = tok.encode(text)
    compression_ratio = len(text.encode("utf-8")) / max(1, len(token_ids))

    longest_token_bytes = max((len(b) for b in vocab.values()), default=0)

    start = time.time()
    _ = tok.encode(text)
    throughput = len(token_ids) / max(1e-9, time.time() - start)

    stats = {
        "input_path": str(input_path),
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "num_merges": len(merges),
        "train_wall_clock_sec": t.elapsed(),
        "compression_ratio_bytes_per_token": compression_ratio,
        "longest_token_bytes": longest_token_bytes,
        "encode_throughput_tokens_per_sec": throughput,
    }
    save_json(Path(args.output_dir) / "stats.json", stats)
    if progress:
        print("[train_tokenizer] done", flush=True)


if __name__ == "__main__":
    main()
