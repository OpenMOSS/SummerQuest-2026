"""顺序运行移除 RMSNorm 的两个学习率消融实验。"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_with_host_nvidia.sh"
CONFIGS = ("TS_no_rmsnorm_lr1e3", "TS_no_rmsnorm_lr3e4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="实际顺序训练；默认仅打印命令")
    args = parser.parse_args()
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive")

    for index, config_name in enumerate(CONFIGS, start=1):
        command = [
            str(LAUNCHER),
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            "scripts/train_lm.py",
            "--config-name",
            config_name,
        ]
        print(f"[{index}/{len(CONFIGS)}] {config_name}")
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)

    if not args.execute:
        print("\n这是 dry run；确认计划后加 --execute 才会开始训练。")


if __name__ == "__main__":
    main()
