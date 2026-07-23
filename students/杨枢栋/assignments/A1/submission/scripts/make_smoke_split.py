from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--valid-bytes", type=int, default=4096)
    args = parser.parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    valid = text[: args.valid_bytes]
    train = text[args.valid_bytes :]
    if len(train) < args.valid_bytes:
        midpoint = max(1, len(text) // 5)
        valid = text[:midpoint]
        train = text[midpoint:]
    (out_dir / "train.txt").write_text(train, encoding="utf-8")
    (out_dir / "valid.txt").write_text(valid, encoding="utf-8")
    print(f"wrote {out_dir / 'train.txt'} and {out_dir / 'valid.txt'}")


if __name__ == "__main__":
    main()
