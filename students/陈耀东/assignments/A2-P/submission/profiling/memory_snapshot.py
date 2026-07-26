"""生成单个 A2-P PyTorch allocator snapshot。

该入口复用统一 benchmark 的 ``--memory-snapshot`` 参数。snapshot 是本地
抽查证据，不能进入公开提交目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one A2-P memory snapshot.")
    parser.add_argument("--model-size", choices=("large", "xl"), default="xl")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--mode", choices=("inference", "train_step"), default="inference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "profiling.benchmark",
        "--device",
        "cuda",
        "--model-size",
        args.model_size,
        "--context-length",
        str(args.context_length),
        "--batch-size",
        "1",
        "--dtype",
        "fp32",
        "--mode",
        args.mode,
        "--warmup",
        "2",
        "--repeats",
        "1",
        "--memory-snapshot",
        str(args.output),
        "--allow-oom",
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
