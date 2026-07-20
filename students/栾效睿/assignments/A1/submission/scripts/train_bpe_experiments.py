from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.train_bpe import run_train_bpe


DEFAULT_CONFIG = ROOT / "configs" / "train_bpe_experiments.json"


def repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def max_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_display(token: bytes) -> str:
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return token.hex()


def longest_tokens(vocab: dict[int, bytes]) -> list[dict[str, Any]]:
    max_len = max((len(token) for token in vocab.values()), default=0)
    rows = [
        {"id": token_id, "byte_length": len(token), "text": token_display(token), "hex": token.hex()}
        for token_id, token in vocab.items()
        if len(token) == max_len
    ]
    return sorted(rows, key=lambda item: item["id"])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_one(run_cfg: dict[str, Any], global_cfg: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    input_path = repo_path(run_cfg["input_path"])
    output_dir = repo_path(run_cfg["output_dir"])
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input corpus: {rel_path(input_path)}")

    workers = int(global_cfg["workers"])
    target_batch_bytes = int(global_cfg["target_batch_bytes"])
    start = time.perf_counter()
    started_at = now_utc()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=int(run_cfg["vocab_size"]),
        special_tokens=list(global_cfg["special_tokens"]),
        output_dir=output_dir,
        workers=workers,
        target_batch_bytes=target_batch_bytes,
    )
    wall_clock_sec = time.perf_counter() - start
    ended_at = now_utc()

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    record = {
        "run_name": run_cfg["name"],
        "dataset": run_cfg["dataset"],
        "status": "completed",
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_clock_sec": wall_clock_sec,
        "input_path": rel_path(input_path),
        "input_bytes": input_path.stat().st_size,
        "vocab_size_requested": int(run_cfg["vocab_size"]),
        "vocab_size_actual": len(vocab),
        "merge_count": len(merges),
        "special_tokens": list(global_cfg["special_tokens"]),
        "workers": workers,
        "target_batch_bytes": target_batch_bytes,
        "output_dir": rel_path(output_dir),
        "vocab_path": rel_path(vocab_path),
        "merges_path": rel_path(merges_path),
        "vocab_sha256": sha256_file(vocab_path),
        "merges_sha256": sha256_file(merges_path),
        "max_rss_bytes": max_rss_bytes(),
        "longest_tokens": longest_tokens(vocab),
    }
    write_json(log_dir / f"{run_cfg['name']}.json", record)
    append_jsonl(log_dir / "results.jsonl", record)
    print(
        f"{run_cfg['name']}: merges={len(merges)} "
        f"time={wall_clock_sec:.2f}s max_rss={record['max_rss_bytes'] / 1e9:.2f}GB",
        flush=True,
    )
    return record


def load_existing_runs(
    runs: list[dict[str, Any]],
    log_dir: Path,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    records = []
    for run in runs:
        path = log_dir / f"{run['name']}.json"
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(f"Missing run log: {rel_path(path)}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def write_summary(config_path: Path, log_dir: Path, records: list[dict[str, Any]]) -> None:
    write_json(
        log_dir / "summary.json",
        {
            "config_path": rel_path(config_path),
            "log_dir": rel_path(log_dir),
            "runs": records,
        },
    )
    results_path = log_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    for record in records:
        append_jsonl(results_path, record)
    print(f"wrote {rel_path(log_dir / 'summary.json')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible BPE training experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", choices=["tinystories", "owt"], help="Run only one configured BPE experiment.")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summary.json/results.jsonl from existing per-run logs without retraining.",
    )
    args = parser.parse_args()

    config_path = repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    log_dir = repo_path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    selected = [run for run in config["runs"] if args.only is None or run["name"] == args.only]
    records = load_existing_runs(config["runs"], log_dir) if args.summarize_only else [
        run_one(run, config, log_dir) for run in selected
    ]
    if args.only and not args.summarize_only:
        records = load_existing_runs(config["runs"], log_dir, allow_missing=True)
    write_summary(config_path, log_dir, records)


if __name__ == "__main__":
    main()
