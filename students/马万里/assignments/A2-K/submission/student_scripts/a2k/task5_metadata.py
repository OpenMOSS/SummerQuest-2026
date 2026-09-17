from __future__ import annotations

import argparse
import re
import json
import subprocess
from pathlib import Path

import torch

NVIDIA_SMI_QUERY = "name,memory.total,memory.free,driver_version,power.limit,pstate"
NVIDIA_SMI_TIMEOUT_SECONDS = 10


def nvidia_smi_snapshot() -> dict:
    """采集脱敏的 nvidia-smi 字段；工具缺失或查询失败时返回空字典。

    字段不可用时 nvidia-smi 会输出 [N/A]，这里直接跳过而不是写进报告。
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    lines = [line.strip() for line in completed.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return {}
    keys = ("name", "memory_total_mib", "memory_free_mib", "driver_version", "power_limit_w", "pstate")
    values = [field.strip() for field in lines[0].split(",")]
    snapshot = {}
    for key, raw in zip(keys, values):
        if not raw or raw.upper() in {"[N/A]", "N/A"}:
            continue
        if key in {"memory_total_mib", "memory_free_mib", "power_limit_w"}:
            try:
                snapshot[key] = float(raw.split()[0])
            except ValueError:
                continue
        else:
            snapshot[key] = raw
    return snapshot


#: 内部调度资源名不应出现在公开产物里；写入 metadata 前统一做一次兜底替换。
_SCHEDULER_PATTERN = re.compile(r"\bsrun\b[^&|;]*?(?=(?:&&|\|\||;|$))")


def public_command(command: str) -> str:
    """把可能带内部调度前缀的命令改写为公开的单卡执行说明。"""
    if not _SCHEDULER_PATTERN.search(command):
        return command
    stripped = _SCHEDULER_PATTERN.sub("", command).strip()
    return f"在单张 RTX 4090 上执行：{stripped}" if stripped else "在单张 RTX 4090 上执行"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--command",
        default=(
            "python student_scripts/a2k/task5_metadata.py && "
            "python student_scripts/a2k/task5_correctness.py --length 128 512 2048 && "
            "python student_scripts/a2k/flash_benchmark.py --implementation eager "
            "--sequence-length 512 --head-dim 64 --phase forward"
        ),
        help="写入 metadata 的复现命令（不要包含内部调度资源名称）",
    )
    parser.add_argument("--commit", default="ca8bc81a59b70516f7ebb2da4808daade877c736")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        metadata = {"status": "skip", "error": "CUDA unavailable",
                    "command": public_command(args.command), "commit": args.commit, "seed": args.seed}
    else:
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        try:
            import triton

            triton_version = triton.__version__
        except ImportError:
            triton_version = None
        metadata = {
            "gpu": torch.cuda.get_device_name(0),
            "total_memory_mib": props.total_memory / 2**20,
            "cuda_reported_total_mib": total_bytes / 2**20,
            "free_memory_at_start_mib": free_bytes / 2**20,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "triton": triton_version,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allocator_limit_mib": 23552,
            "allocator_fraction": min(1.0, 23552 / (props.total_memory / 2**20)),
            "hard_limit_mib": 24576,
            "nvidia_smi": nvidia_smi_snapshot() or None,
            "timer": "triton.testing.do_bench(warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])",
            "warmup_ms": 100,
            "rep_ms": 300,
            "quantiles": [0.2, 0.5, 0.8],
            "dtype": "bfloat16",
            "is_causal": True,
            "batch_size": 1,
            "memory_measure_iterations": 5,
            "seed": args.seed,
            "command": public_command(args.command),
            "commit": args.commit,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
