"""Generate validated experiment configurations for section 7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cs336_basics.training import TrainingConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"
LR_VALUES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def load_base(name: str) -> dict[str, Any]:
    return json.loads((CONFIGS_DIR / f"{name}_full.json").read_text(encoding="utf-8"))


def _identity(config: dict[str, Any], name: str) -> dict[str, Any]:
    config["output_dir"] = f"data/runs/{name}"
    config["wandb_run_name"] = name
    return config


def _with_lr(config: dict[str, Any], learning_rate: float) -> None:
    config["max_learning_rate"] = learning_rate
    config["min_learning_rate"] = learning_rate / 10


def learning_rate_run_name(learning_rate: float) -> str:
    scientific = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"lr_{scientific}"


def build_lr_sweep(base: dict[str, Any]) -> list[dict[str, Any]]:
    configs = []
    for learning_rate in LR_VALUES:
        config = dict(base)
        _with_lr(config, learning_rate)
        config.update(max_iters=1000, warmup_iters=50, eval_interval=50, checkpoint_interval=1000)
        configs.append(_identity(config, learning_rate_run_name(learning_rate)))
    return configs


def build_baseline(base: dict[str, Any], learning_rate: float) -> list[dict[str, Any]]:
    config = dict(base)
    _with_lr(config, learning_rate)
    return [_identity(config, "tinystories_baseline")]


def select_batch_sizes(max_batch: int) -> list[int]:
    if max_batch < 128:
        raise ValueError("max_batch must be at least 128 for the assignment experiment")
    midpoint = round(((128 + max_batch) / 2) / 32) * 32
    values = {1, 64, 128, midpoint, max_batch}
    for candidate in (32, 96, 160, 192, 256, 384, 512):
        if len(values) == 5:
            break
        if candidate <= max_batch:
            values.add(candidate)
    result = sorted(values)
    if len(result) != 5:
        raise ValueError(f"could not construct five unique batch sizes for max_batch={max_batch}")
    return result


def build_batch_size(base: dict[str, Any], learning_rate: float, max_batch: int) -> list[dict[str, Any]]:
    configs = []
    for batch_size in select_batch_sizes(max_batch):
        config = dict(base)
        _with_lr(config, learning_rate)
        config["batch_size"] = batch_size
        configs.append(_identity(config, f"batch_{batch_size}"))
    return configs


def build_ablations(base: dict[str, Any], learning_rate: float) -> list[dict[str, Any]]:
    variants = [
        ("ablation_no_rmsnorm", {"norm_mode": "none"}, learning_rate),
        ("ablation_no_rmsnorm_low_lr", {"norm_mode": "none"}, learning_rate / 3),
        ("ablation_post_norm", {"norm_mode": "post"}, learning_rate),
        ("ablation_nope", {"use_rope": False}, learning_rate),
        ("ablation_silu", {"ffn_type": "silu", "d_ff": 2048}, learning_rate),
    ]
    configs = []
    for name, overrides, variant_lr in variants:
        config = dict(base)
        _with_lr(config, variant_lr)
        config.update(overrides)
        configs.append(_identity(config, name))
    return configs


def build_owt(base: dict[str, Any], learning_rate: float) -> list[dict[str, Any]]:
    config = dict(base)
    _with_lr(config, learning_rate)
    return [_identity(config, "owt_baseline")]


def build_leaderboard(base: dict[str, Any], learning_rate: float) -> list[dict[str, Any]]:
    config = dict(base)
    _with_lr(config, learning_rate)
    config.update(tied_embeddings=True, max_train_seconds=2640, eval_interval=50, checkpoint_interval=250)
    return [_identity(config, "leaderboard")]


def build_smoke(base: dict[str, Any], max_iters: int) -> list[dict[str, Any]]:
    if max_iters <= 0:
        raise ValueError("smoke max_iters must be positive")
    config = dict(base)
    config.update(
        batch_size=8,
        max_iters=max_iters,
        warmup_iters=min(5, max_iters),
        eval_interval=min(10, max_iters),
        eval_iters=2,
        log_interval=min(5, max_iters),
        checkpoint_interval=min(20, max_iters),
        compile_model=False,
        wandb_project=None,
        max_train_seconds=None,
    )
    return [_identity(config, "smoke_test")]


def build_divergence_probe(base: dict[str, Any], learning_rate: float, max_iters: int = 1000) -> list[dict[str, Any]]:
    if learning_rate <= 0:
        raise ValueError("divergence learning rate must be positive")
    if max_iters <= 0:
        raise ValueError("divergence max_iters must be positive")
    config = dict(base)
    _with_lr(config, learning_rate)
    config.update(
        max_iters=max_iters,
        warmup_iters=min(50, max_iters),
        eval_interval=min(50, max_iters),
        checkpoint_interval=max_iters,
        compile_model=False,
    )
    name = f"divergence_{learning_rate_run_name(learning_rate)}"
    return [_identity(config, name)]


def build_all(
    ts_base: dict[str, Any],
    owt_base: dict[str, Any],
    learning_rate: float,
    owt_lr: float,
    leaderboard_lr: float,
    max_batch: int,
) -> list[dict[str, Any]]:
    configs = (
        build_lr_sweep(ts_base)
        + build_baseline(ts_base, learning_rate)
        + build_batch_size(ts_base, learning_rate, max_batch)
        + build_ablations(ts_base, learning_rate)
        + build_owt(owt_base, owt_lr)
        + build_leaderboard(owt_base, leaderboard_lr)
    )
    validate_configs(configs, expected_count=18)
    return configs


def validate_configs(configs: list[dict[str, Any]], expected_count: int | None = None) -> None:
    if expected_count is not None and len(configs) != expected_count:
        raise ValueError(f"expected {expected_count} configs, got {len(configs)}")
    names = [str(config["wandb_run_name"]) for config in configs]
    if len(names) != len(set(names)):
        raise ValueError("run names must be unique")
    for config in configs:
        name = str(config["wandb_run_name"])
        if Path(str(config["output_dir"])).name != name:
            raise ValueError(f"output_dir and run name disagree for {name}")
        TrainingConfig(**config).validate()


def write_configs(configs: list[dict[str, Any]]) -> list[Path]:
    validate_configs(configs)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    names = {str(config["wandb_run_name"]) for config in configs}
    for prefix in ("lr_", "batch_"):
        if any(name.startswith(prefix) for name in names):
            for stale_path in EXPERIMENTS_DIR.glob(f"{prefix}*.json"):
                if stale_path.stem not in names:
                    stale_path.unlink()
    paths = []
    for config in configs:
        path = EXPERIMENTS_DIR / f"{config['wandb_run_name']}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["lr_sweep", "baseline", "batch_size", "ablations", "owt", "leaderboard", "all", "smoke", "divergence"],
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--owt-lr", type=float, default=3e-4)
    parser.add_argument("--leaderboard-lr", type=float, default=3e-4)
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--smoke-iters", type=int, default=20)
    parser.add_argument("--probe-iters", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts_base = load_base("tinystories")
    owt_base = load_base("owt")
    builders = {
        "lr_sweep": lambda: build_lr_sweep(ts_base),
        "baseline": lambda: build_baseline(ts_base, args.lr),
        "batch_size": lambda: build_batch_size(ts_base, args.lr, args.max_batch),
        "ablations": lambda: build_ablations(ts_base, args.lr),
        "owt": lambda: build_owt(owt_base, args.owt_lr),
        "leaderboard": lambda: build_leaderboard(owt_base, args.leaderboard_lr),
        "all": lambda: build_all(ts_base, owt_base, args.lr, args.owt_lr, args.leaderboard_lr, args.max_batch),
        "smoke": lambda: build_smoke(ts_base, args.smoke_iters),
        "divergence": lambda: build_divergence_probe(ts_base, args.lr, args.probe_iters),
    }
    write_configs(builders[args.command]())


if __name__ == "__main__":
    main()
