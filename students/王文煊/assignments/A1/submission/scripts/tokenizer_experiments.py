"""Tokenizer experiments: compression ratio, cross-tokenization, throughput.

Samples documents (split on <|endoftext|>) from each corpus, encodes them with
the matching tokenizer, and reports bytes/token. Also cross-encodes the OWT
sample with the TinyStories tokenizer, and measures encode throughput.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return Tokenizer(data["vocab"], data["merges"], data["special_tokens"])


def sample_docs(path: str, n: int) -> list[str]:
    """Read the first ``n`` <|endoftext|>-delimited documents from a file."""
    docs: list[str] = []
    buf: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if EOT in line:
                pre, _, post = line.partition(EOT)
                buf.append(pre)
                docs.append("".join(buf))
                buf = [post]
                if len(docs) >= n:
                    break
            else:
                buf.append(line)
    return [d for d in docs if d.strip()][:n]


def compression_ratio(tok: Tokenizer, docs: list[str]) -> tuple[float, list[float]]:
    ratios = []
    tot_bytes = tot_tokens = 0
    for d in docs:
        nb = len(d.encode("utf-8"))
        nt = len(tok.encode(d))
        if nt:
            ratios.append(nb / nt)
            tot_bytes += nb
            tot_tokens += nt
    return (tot_bytes / tot_tokens if tot_tokens else 0.0), ratios


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ts-tokenizer", default="artifacts/tinystories_tokenizer.pkl")
    p.add_argument("--owt-tokenizer", default="artifacts/owt_tokenizer.pkl")
    p.add_argument("--ts-data", default="data/TinyStoriesV2-GPT4-valid.txt")
    p.add_argument("--owt-data", default="data/owt_valid.txt")
    p.add_argument("--num-docs", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ts_tok = load_tokenizer(args.ts_tokenizer)
    owt_tok = load_tokenizer(args.owt_tokenizer)

    ts_docs = sample_docs(args.ts_data, args.num_docs)
    owt_docs = sample_docs(args.owt_data, args.num_docs)

    ts_cr, ts_list = compression_ratio(ts_tok, ts_docs)
    owt_cr, owt_list = compression_ratio(owt_tok, owt_docs)
    cross_cr, cross_list = compression_ratio(ts_tok, owt_docs)  # OWT text with TS tokenizer

    # Throughput measured on the concatenated OWT sample with the OWT tokenizer.
    sample_text = "".join(owt_docs)
    n_bytes = len(sample_text.encode("utf-8"))
    start = time.time()
    n_tokens = len(owt_tok.encode(sample_text))
    elapsed = time.time() - start
    throughput = n_bytes / elapsed if elapsed else 0.0
    pile_bytes = 825 * (1024**3)
    pile_hours = pile_bytes / throughput / 3600 if throughput else 0.0

    result = {
        "tinystories_compression_ratio": round(ts_cr, 4),
        "owt_compression_ratio": round(owt_cr, 4),
        "owt_with_tinystories_tokenizer_ratio": round(cross_cr, 4),
        "per_doc_ratios": {
            "tinystories": [round(r, 3) for r in ts_list],
            "owt": [round(r, 3) for r in owt_list],
            "owt_cross": [round(r, 3) for r in cross_list],
        },
        "throughput_bytes_per_sec": round(throughput, 1),
        "throughput_tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed else 0.0,
        "pile_825GB_estimate_hours": round(pile_hours, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
