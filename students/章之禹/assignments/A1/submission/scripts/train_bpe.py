#!/usr/bin/env python3
"""Train and serialize a byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import json
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import psutil

from cs336_basics.bpe import save_tokenizer_files, train_bpe


class _ProcessTreeMemorySampler:
    """Sample aggregate RSS for this process and all of its descendants."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        if interval_seconds <= 0:
            raise ValueError("memory sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self.parent = psutil.Process()
        self.peak_parent_rss_bytes = 0
        self.peak_process_tree_rss_bytes = 0
        self.sample_count = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bpe-memory-sampler", daemon=True)

    @staticmethod
    def _rss(process: psutil.Process) -> int:
        try:
            return int(process.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            return 0

    def _sample(self) -> None:
        parent_rss = self._rss(self.parent)
        try:
            children = self.parent.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            children = []
        process_tree_rss = parent_rss + sum(self._rss(child) for child in children)
        self.peak_parent_rss_bytes = max(self.peak_parent_rss_bytes, parent_rss)
        self.peak_process_tree_rss_bytes = max(self.peak_process_tree_rss_bytes, process_tree_rss)
        self.sample_count += 1

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        self._sample()


def _config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("rb") as config_file:
        payload = tomllib.load(config_file)
    return payload.get("tokenizer", payload.get("bpe", payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", nargs="?", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _config(args.config)

    input_path = args.input_path or (Path(config["input_path"]) if "input_path" in config else None)
    vocab_size = args.vocab_size if args.vocab_size is not None else config.get("vocab_size")
    special_tokens = args.special_tokens if args.special_tokens is not None else config.get("special_tokens", [])
    num_workers = args.num_workers if args.num_workers is not None else config.get("num_workers", 1)
    output_dir = args.output_dir or Path(config.get("output_dir", "artifacts/tokenizer"))
    if input_path is None or vocab_size is None:
        parser.error("input_path and --vocab-size (or their config values) are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    memory_sampler = _ProcessTreeMemorySampler()
    memory_sampler.start()
    try:
        vocab, merges = train_bpe(
            input_path,
            int(vocab_size),
            list(special_tokens),
            num_workers=int(num_workers),
        )
    finally:
        memory_sampler.stop()
    elapsed = time.perf_counter() - started
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    save_tokenizer_files(vocab, merges, vocab_path, merges_path)

    longest = max(vocab.values(), key=lambda token: (len(token), token), default=b"")
    metrics = {
        "elapsed_seconds": elapsed,
        "merge_count": len(merges),
        "peak_parent_rss_bytes": memory_sampler.peak_parent_rss_bytes,
        "peak_process_tree_rss_bytes": memory_sampler.peak_process_tree_rss_bytes,
        "memory_sample_count": memory_sampler.sample_count,
        "memory_sampling_interval_seconds": memory_sampler.interval_seconds,
        "memory_measurement": "maximum sampled aggregate RSS of the parent and all recursive child processes",
        "vocab_size": len(vocab),
        "longest_token_bytes": len(longest),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
