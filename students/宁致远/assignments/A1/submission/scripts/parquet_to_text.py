"""Concatenate OWT parquet shards into a text file with <|endoftext|> between docs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sep", default="<|endoftext|>")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    print(f"found {len(paths)} parquet files")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out, "w", encoding="utf-8") as f:
        for p in paths:
            t = pq.read_table(p, columns=["text"])
            texts = t.column("text").to_pylist()
            for doc in texts:
                f.write(doc)
                f.write("\n" + args.sep + "\n")
                total += len(doc.encode("utf-8"))
            print(f"  wrote {len(texts)} docs from {Path(p).name}; cumulative bytes {total:,}")
    print(f"done: {out} = {total:,} bytes")


if __name__ == "__main__":
    main()
