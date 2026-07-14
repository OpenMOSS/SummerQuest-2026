#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


def run_one(config_path: Path, gpu_id: int, train_data: str, val_data: str, output_root: Path) -> dict:
    run_name = config_path.stem
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").is_file():
        return {
            "run": run_name,
            "gpu_id": gpu_id,
            "returncode": 0,
            "elapsed_sec": 0.0,
            "output_dir": str(output_dir),
            "skipped": True,
        }
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        sys.executable,
        "scripts/train_lm.py",
        "--config",
        str(config_path),
        "--train-data",
        train_data,
        "--val-data",
        val_data,
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda:0",
    ]
    start = time.time()
    with open(output_dir / "stdout.log", "w", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "run": run_name,
        "gpu_id": gpu_id,
        "returncode": process.returncode,
        "elapsed_sec": time.time() - start,
        "output_dir": str(output_dir),
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short A1 experiments concurrently on at most eight devices.")
    parser.add_argument("--config-dir", default="artifacts/experiment_configs")
    parser.add_argument("--train-data", default="data/tinystories_train.bin")
    parser.add_argument("--val-data", default="data/tinystories_valid.bin")
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--max-gpus", type=int, default=8)
    args = parser.parse_args()

    if not 1 <= args.max_gpus <= 8:
        raise ValueError("max-gpus must be between 1 and 8")
    available = torch.cuda.device_count()
    num_gpus = min(args.max_gpus, available)
    if num_gpus < 1:
        raise RuntimeError("no CUDA-compatible device is visible")
    config_paths = sorted(Path(args.config_dir).glob("*.json"))
    if not config_paths:
        raise FileNotFoundError(f"no experiment configurations found in {args.config_dir}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu_id in range(num_gpus):
        gpu_pool.put(gpu_id)

    def run_with_available_gpu(config_path: Path) -> dict:
        gpu_id = gpu_pool.get()
        try:
            return run_one(config_path, gpu_id, args.train_data, args.val_data, output_root)
        finally:
            gpu_pool.put(gpu_id)

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(run_with_available_gpu, config_path): config_path for config_path in config_paths
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)

    summary = {"num_gpus": num_gpus, "runs": sorted(results, key=lambda result: result["run"])}
    (output_root / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if any(result["returncode"] != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
