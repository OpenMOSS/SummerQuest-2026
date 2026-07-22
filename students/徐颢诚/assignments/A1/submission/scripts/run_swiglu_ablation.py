"""运行参数量近似匹配的 SwiGLU-vs-SiLU 消融实验。"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_with_host_nvidia.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="实际训练；默认只打印命令")
    args = parser.parse_args()
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive")
    command = [
        str(LAUNCHER),
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "scripts/train_lm.py",
        "--config-name",
        "TS_silu",
    ]
    print(" ".join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
    else:
        print("\n这是 dry run；加 --execute 后开始训练。")


if __name__ == "__main__":
    main()
