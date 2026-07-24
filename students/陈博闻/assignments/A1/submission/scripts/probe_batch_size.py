from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_probe(args: argparse.Namespace, batch_size: int) -> dict[str, object]:
    log_path = Path(args.output_dir) / f"batch_{batch_size}.jsonl"
    summary_path = Path(args.output_dir) / f"batch_{batch_size}_summary.json"
    cmd = [
        sys.executable,
        "scripts/train_lm.py",
        "--train-data",
        args.train_data,
        "--val-data",
        args.val_data,
        "--vocab-size",
        str(args.vocab_size),
        "--context-length",
        str(args.context_length),
        "--d-model",
        str(args.d_model),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--d-ff",
        str(args.d_ff),
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(min(batch_size, args.eval_batch_size)),
        "--max-iters",
        str(args.steps),
        "--eval-iters",
        str(args.eval_iters),
        "--eval-every",
        str(args.steps),
        "--log-every",
        "1",
        "--device",
        args.device,
        "--log-jsonl",
        str(log_path),
        "--summary-json",
        str(summary_path),
        "--run-name",
        f"batch_probe_{batch_size}",
    ]
    result = subprocess.run(cmd, cwd=args.repo_root, text=True, capture_output=True)
    record: dict[str, object] = {
        "batch_size": batch_size,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
    if summary_path.exists():
        record["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe which batch sizes fit for a model/data setup.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--eval-iters", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [run_probe(args, batch_size) for batch_size in args.batch_sizes]
    summary_path = output_dir / "batch_probe_summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    main()
