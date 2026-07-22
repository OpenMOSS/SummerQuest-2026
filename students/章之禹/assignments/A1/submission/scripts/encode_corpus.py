#!/usr/bin/env python3
"""Encode a UTF-8 corpus into a memory-mappable NumPy uint16 array."""

from __future__ import annotations

import argparse
import array
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def _config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("rb") as config_file:
        payload = tomllib.load(config_file)
    return payload.get("encoding", payload.get("tokenizer", payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", nargs="?", type=Path)
    parser.add_argument("output_path", nargs="?", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--merges", type=Path)
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    parser.add_argument("--batch-tokens", type=int, default=1_000_000)
    args = parser.parse_args()
    config = _config(args.config)

    input_path = args.input_path or (Path(config["input_path"]) if "input_path" in config else None)
    output_path = args.output_path or (Path(config["output_path"]) if "output_path" in config else None)
    vocab_path = args.vocab or (Path(config["vocab_path"]) if "vocab_path" in config else None)
    merges_path = args.merges or (Path(config["merges_path"]) if "merges_path" in config else None)
    special_tokens = args.special_tokens if args.special_tokens is not None else config.get("special_tokens", [])
    if None in (input_path, output_path, vocab_path, merges_path):
        parser.error("input_path, output_path, --vocab, and --merges are required")
    assert input_path is not None and output_path is not None and vocab_path is not None and merges_path is not None
    if output_path.suffix != ".npy":
        parser.error("output_path must end in .npy")
    if max(args.batch_tokens, 0) == 0:
        parser.error("--batch-tokens must be positive")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, list(special_tokens))
    if not tokenizer.vocab or min(tokenizer.vocab) < 0 or max(tokenizer.vocab) >= 2**16:
        raise ValueError("all token IDs must fit in uint16")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    batch = array.array("H")
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.raw.", delete=False) as raw_file:
        raw_path = Path(raw_file.name)
        array_path: Path | None = None
        try:
            with input_path.open(encoding="utf-8") as corpus:
                for token_id in tokenizer.encode_iterable(corpus):
                    if not 0 <= token_id < 2**16:
                        raise ValueError(f"token ID {token_id} does not fit in uint16")
                    batch.append(token_id)
                    count += 1
                    if len(batch) >= args.batch_tokens:
                        batch.tofile(raw_file)
                        batch = array.array("H")
            if batch:
                batch.tofile(raw_file)
            raw_file.flush()

            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.name}.array.",
                suffix=".npy",
                delete=False,
            ) as array_file:
                array_path = Path(array_file.name)
            encoded = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint16, shape=(count,))
            if count:
                raw = np.memmap(raw_path, mode="r", dtype=np.uint16, shape=(count,))
                encoded[:] = raw
                encoded.flush()
                del raw
            del encoded
            os.replace(array_path, output_path)
            array_path = None
        finally:
            os.unlink(raw_path)
            if array_path is not None:
                array_path.unlink(missing_ok=True)

    metrics = {"dtype": "uint16", "output_path": str(output_path), "token_count": count}
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
