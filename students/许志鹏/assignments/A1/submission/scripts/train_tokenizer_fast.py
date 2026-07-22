from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_BPE_DIR = REPO_ROOT / "scripts" / "fast_bpe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE using the Rust fast_bpe helper.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--release", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_binary(release: bool) -> Path:
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    subprocess.run(cmd, cwd=FAST_BPE_DIR, check=True)
    profile = "release" if release else "debug"
    exe = FAST_BPE_DIR / "target" / profile / ("fast_bpe.exe" if sys.platform == "win32" else "fast_bpe")
    if not exe.exists():
        raise FileNotFoundError(exe)
    return exe


def load_fast_json(path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = {int(idx): bytes.fromhex(token_hex) for idx, token_hex in payload["vocab"]}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in payload["merges"]]
    return vocab, merges, payload.get("special_tokens", [])


def main() -> None:
    args = parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    exe = build_binary(args.release)
    with tempfile.TemporaryDirectory() as tmpdir:
        fast_json = Path(tmpdir) / "tokenizer.json"
        cmd = [
            str(exe),
            "--input",
            args.input,
            "--output-json",
            str(fast_json),
            "--vocab-size",
            str(args.vocab_size),
            "--progress-interval",
            str(args.progress_interval),
            "--num-workers",
            str(args.num_workers),
        ]
        for token in special_tokens:
            cmd.extend(["--special-token", token])
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        vocab, merges, special_tokens = load_fast_json(fast_json)

    with output.open("wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, f)

    metadata = {
        "input": str(args.input),
        "output": str(output),
        "vocab_size": args.vocab_size,
        "special_tokens": special_tokens,
        "num_merges": len(merges),
        "backend": "rust-fast_bpe",
        "num_workers": args.num_workers,
        "elapsed_sec": time.time() - start,
    }
    if args.metadata is not None:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
