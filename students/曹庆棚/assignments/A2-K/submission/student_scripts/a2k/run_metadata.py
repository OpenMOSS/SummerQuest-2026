from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch

try:
    import triton
    _TRITON_VERSION = triton.__version__
except Exception:
    _TRITON_VERSION = None

from .common import allocator_guard, environment_metadata, git_commit, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Capture redacted A2-K environment metadata")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    # Correctness runs use FP32 with both TF32 paths disabled.  Performance
    # runs are BF16 and are unaffected by this setting.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    guard = allocator_guard()
    meta = {
        "assignment": "A2-K", "seed": args.seed, "git_commit": git_commit(),
        "commands": {
            "metadata": "python -m student_scripts.a2k.run_metadata --output results/run_metadata.json",
            "correctness": "python -m student_scripts.a2k.run_correctness --output results/correctness.json --seeds 0,1,2",
            "checkpoint": "python -m student_scripts.a2k.benchmark_checkpointing --output results/checkpointing.csv --seqs 1024,2048 --block-sizes 0,1,2,4,8 --warmup 3 --steps 5",
            "attention": "python -m student_scripts.a2k.benchmark_attention --output results/attention_baseline.csv --seqs 512,2048,8192 --dims 64,128 --warmup 100 --rep 300",
            "compile": "python -m student_scripts.a2k.benchmark_compile --output results/compile_comparison.csv --warmup 100 --rep 300",
            "flash_core": "python -m student_scripts.a2k.benchmark_flash --output results/flash_benchmark.csv --seqs 512,2048,8192 --dims 64,128 --implementations eager,compiled,triton --warmup 100 --rep 300 --query-tile 32 --key-tile 64 --num-warps 4 --num-stages 1",
            "flash_boundary": "python -m student_scripts.a2k.benchmark_flash --output results/flash_benchmark.csv --seqs 16384 --dims 64,128 --implementations eager,triton --warmup 100 --rep 300 --query-tile 32 --key-tile 64 --num-warps 4 --num-stages 1"
        },
        "triton_version": _TRITON_VERSION,
        "compiled_mode": "reduce-overhead",
        "timer": "time.perf_counter with torch.cuda.synchronize before and after each sample",
        "quantiles": [0.2, 0.5, 0.8],
        "attention_warmup": 100,
        "attention_measurement_rep": 300,
        "flash_warmup": 100,
        "flash_measurement_rep": 300,
        "checkpoint_warmup_steps": 3,
        "checkpoint_measurement_steps": 5,
        **environment_metadata(), **guard
    }
    try:
        fields = "name,memory.total,memory.free,driver_version,power.limit,pstate"
        raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], text=True).strip()
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) >= 6:
            meta["nvidia_smi"] = {"gpu_name": parts[0], "memory_total": parts[1], "memory_free": parts[2], "driver_version": parts[3], "power_limit": parts[4], "pstate": parts[5]}
    except Exception:
        meta["nvidia_smi"] = {"status": "unavailable"}
    # Do not emit hostnames, usernames, UUIDs, or paths.
    write_json(args.output, meta)
    print(meta)


if __name__ == "__main__":
    main()
