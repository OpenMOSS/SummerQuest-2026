from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scripts.tokenizer_io import load_tokenizer
from scripts.utils import Timer, load_json, save_json, tqdm_disabled


def _iter_text_with_progress(path: str | os.PathLike, *, desc: str):
    total = os.path.getsize(path)
    with open(path, "rb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=desc,
        disable=tqdm_disabled(),
    ) as pbar:
        for raw in f:
            pbar.update(len(raw))
            yield raw.decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    special_tokens = list(cfg.get("special_tokens", []))
    tokenizer = load_tokenizer(args.tokenizer_dir, special_tokens=special_tokens)

    t = Timer.start_now()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    n_tokens = 0
    for token_id in tokenizer.encode_iterable(_iter_text_with_progress(args.input_path, desc="count tokens")):
        n_tokens += 1

    dtype = np.uint16 if int(cfg.get("vocab_size", 0)) <= 65535 else np.uint32
    arr = np.lib.format.open_memmap(args.output_path, mode="w+", dtype=dtype, shape=(n_tokens,))

    i = 0
    for token_id in tokenizer.encode_iterable(_iter_text_with_progress(args.input_path, desc="write tokens")):
        arr[i] = token_id
        i += 1

    arr.flush()

    stats = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "num_tokens": int(n_tokens),
        "dtype": str(arr.dtype),
        "wall_clock_sec": t.elapsed(),
    }
    stats_path = Path(args.output_path).with_suffix(".json")
    save_json(stats_path, stats)


if __name__ == "__main__":
    main()
