#!/usr/bin/env python3
"""Measure tokenizer compression and encoding throughput on a text corpus."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from cs336_basics.tokenizer import Tokenizer


def _config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("rb") as config_file:
        payload = tomllib.load(config_file)
    return payload.get("benchmark", payload.get("tokenizer", payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", nargs="?", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--merges", type=Path)
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    config = _config(args.config)

    input_path = args.input_path or (Path(config["input_path"]) if "input_path" in config else None)
    vocab_path = args.vocab or (Path(config["vocab_path"]) if "vocab_path" in config else None)
    merges_path = args.merges or (Path(config["merges_path"]) if "merges_path" in config else None)
    special_tokens = args.special_tokens if args.special_tokens is not None else config.get("special_tokens", [])
    json_output = args.json_output or (Path(config["json_output"]) if "json_output" in config else None)
    if None in (input_path, vocab_path, merges_path):
        parser.error("input_path, --vocab, and --merges are required")
    assert input_path is not None and vocab_path is not None and merges_path is not None

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, list(special_tokens))
    byte_count = 0

    def measured_chunks():
        nonlocal byte_count
        with input_path.open(encoding="utf-8") as corpus:
            for chunk in corpus:
                byte_count += len(chunk.encode("utf-8"))
                yield chunk

    started = time.perf_counter()
    token_count = sum(1 for _ in tokenizer.encode_iterable(measured_chunks()))
    elapsed = time.perf_counter() - started
    metrics = {
        "bytes": byte_count,
        "tokens": token_count,
        "bytes_per_token": byte_count / token_count if token_count else None,
        "bytes_per_second": byte_count / elapsed if elapsed else None,
        "elapsed_seconds": elapsed,
        "estimated_pile_seconds": 825_000_000_000 / (byte_count / elapsed) if byte_count and elapsed else None,
    }
    rendered = json.dumps(metrics, indent=2) + "\n"
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
