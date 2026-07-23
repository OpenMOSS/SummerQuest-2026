import argparse
import json
import time
from pathlib import Path

import numpy as np

from cs336_basics.bpe_io import load_bpe
from cs336_basics.tokenizer import Tokenizer

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Encode dataset"
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("tokenizer_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--buffer-tokens", type=int, default=1_000_000)

    return parser.parse_args(argv)

def main():
    args = parse_args()

    vocab, merges = load_bpe(args.tokenizer_path)
    tokenizer = Tokenizer(vocab, merges, special_tokens=args.special_token)

    if max(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("token IDs do not fit in uint16")

    input_bytes = args.input_path.stat().st_size
    partial_path = args.output_path.with_name(args.output_path.name + ".partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = []
    token_count = 0
    start = time.perf_counter()

    with (
        open(args.input_path, encoding="utf-8") as source,
        open(partial_path, "wb") as output_file,
    ):
        for token_id in tokenizer.encode_iterable(source):
            buffer.append(token_id)
            if len(buffer) >= args.buffer_tokens:
                np.asarray(buffer, dtype=np.uint16).tofile(output_file)
                token_count += len(buffer)
                buffer.clear()

        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(output_file)
            token_count += len(buffer)
            buffer.clear()

    elapsed = time.perf_counter() - start
    partial_path.replace(args.output_path)

    stats = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "input_bytes": input_bytes,
        "tokens": token_count,
        "elapsed_seconds": elapsed,
        "throughput_mib_per_second": input_bytes / elapsed / (1024 * 1024) if elapsed else 0.0,
        "bytes_per_token": input_bytes / token_count if token_count else 0.0,
        "dtype": "uint16",
    }

    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
