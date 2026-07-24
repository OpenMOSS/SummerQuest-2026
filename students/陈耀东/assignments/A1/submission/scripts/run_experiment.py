"""从单个 JSON 配置启动一次训练实验。

该入口有意不支持一键运行全部实验，避免误触发耗时很长的正式训练。
每次只接受一个配置文件；正式训练前可先使用 --dry-run 检查最终命令。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_KEYS = {"name", "description"}
ARGUMENT_ORDER = (
    "train_data",
    "valid_data",
    "output_dir",
    "resume",
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "rope_theta",
    "normalization",
    "positional_encoding",
    "ffn_type",
    "batch_size",
    "max_iters",
    "max_lr",
    "min_lr",
    "warmup_iters",
    "beta1",
    "beta2",
    "eps",
    "weight_decay",
    "grad_clip",
    "eval_interval",
    "eval_iters",
    "log_interval",
    "checkpoint_interval",
    "device",
    "seed",
    "matmul_precision",
    "compile_mode",
    "amp",
    "fail_on_non_finite",
)
REQUIRED_KEYS = {"train_data", "valid_data", "output_dir", "vocab_size"}


def parse_args() -> argparse.Namespace:
    """解析单实验运行参数。"""
    parser = argparse.ArgumentParser(description="从 JSON 配置启动一次 CS336 训练实验")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="只检查配置并打印命令，不启动训练")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    """读取并校验单次训练配置。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("实验配置顶层必须是 JSON object")

    allowed_keys = set(ARGUMENT_ORDER) | METADATA_KEYS
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise KeyError(f"实验配置包含未知字段：{', '.join(unknown_keys)}")

    missing_keys = sorted(key for key in REQUIRED_KEYS if key not in payload)
    if missing_keys:
        raise KeyError(f"实验配置缺少字段：{', '.join(missing_keys)}")
    return payload


def build_command(config: dict[str, object]) -> list[str]:
    """把 JSON 字段稳定地转换为训练 CLI 参数。"""
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py")]
    for key in ARGUMENT_ORDER:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            raise TypeError(f"配置字段 {key} 不能是 object 或 array")
        command.extend((f"--{key.replace('_', '-')}", str(value)))
    return command


def main() -> None:
    """检查配置并选择打印或启动单次训练。"""
    args = parse_args()
    config = load_config(args.config)
    command = build_command(config)
    display_command = ["python", "scripts/train.py", *command[2:]]

    experiment_name = config.get("name", args.config.stem)
    print(f"experiment={experiment_name}", flush=True)
    print(subprocess.list2cmdline(display_command), flush=True)
    if args.dry_run:
        print("dry_run=True，未启动训练")
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
