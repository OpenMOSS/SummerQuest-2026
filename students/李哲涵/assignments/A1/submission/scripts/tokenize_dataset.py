from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from _project_api import get_tokenizer_cls


SUPPORTED_DTYPES = ("uint16", "uint32", "uint64", "int32", "int64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a text corpus into a 1-D .npy array.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="uint16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.input_path, args.vocab_path, args.merges_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizer_cls = get_tokenizer_cls()
    tokenizer = tokenizer_cls.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        list(args.special_token),
    )

    np_dtype = np.dtype(args.dtype)
    dtype_info = np.iinfo(np_dtype)

    # 先从 vocab 文件验证 id 范围，避免 np.fromiter 在窄 dtype 下发生静默回绕。
    import json

    with args.vocab_path.open("r", encoding="utf-8") as vocab_file:
        serialized_vocab = json.load(vocab_file)
    vocab_ids = [int(token_id) for token_id in serialized_vocab]
    if not vocab_ids:
        raise RuntimeError("vocab 文件为空")
    if min(vocab_ids) < dtype_info.min or max(vocab_ids) > dtype_info.max:
        raise OverflowError(
            f"vocab token id 范围 [{min(vocab_ids)}, {max(vocab_ids)}] "
            f"超出 {args.dtype} 可表示范围"
        )

    with args.input_path.open("r", encoding="utf-8") as text_file:
        token_ids = np.fromiter(tokenizer.encode_iterable(text_file), dtype=np_dtype)

    if token_ids.size == 0:
        raise RuntimeError("tokenizer 没有产生任何 token，请检查输入文件和 encode_iterable")

    max_token_id = int(token_ids.max())
    min_token_id = int(token_ids.min())
    if min_token_id < dtype_info.min or max_token_id > dtype_info.max:
        raise OverflowError(
            f"token id 范围 [{min_token_id}, {max_token_id}] 超出 {args.dtype} 可表示范围"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, token_ids, allow_pickle=False)

    raw_bytes = os.path.getsize(args.input_path)
    bytes_per_token = raw_bytes / token_ids.size
    print(f"tokens={token_ids.size}")
    print(f"dtype={token_ids.dtype} min={min_token_id} max={max_token_id}")
    print(f"raw_bytes={raw_bytes} bytes_per_token={bytes_per_token:.4f}")
    print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
