from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


LR_CANDIDATES = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
THROUGHPUT_BATCHES = (1, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
QUALITY_BATCHES = (16, 32, 64, 128, 256)
WARMUP_RATIOS = (0.0, 0.01, 0.05, 0.10)
QUALITY_TOKEN_BUDGET = 32_768_000
CONTEXT_LENGTH = 256


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def ratio(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparable TinyStories search, selection, final, and ablation configs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("configs/generated"))
    parser.add_argument(
        "--selected-lr",
        type=positive_float,
        default=3e-3,
        help="LR used by batch, warmup, final, and ablation configs after the LR sweep.",
    )
    parser.add_argument(
        "--selected-batch-size",
        type=positive_int,
        default=128,
        help="Batch size used by warmup, final, and ablation configs after the batch sweep.",
    )
    parser.add_argument(
        "--selected-warmup-ratio",
        type=ratio,
        default=0.10,
        help="Warmup ratio used by final and ablation configs after the warmup sweep.",
    )
    return parser.parse_args()


def lr_label(learning_rate: float) -> str:
    mantissa, exponent = f"{learning_rate:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def training_override(
    *,
    steps: int,
    batch_size: int,
    eval_interval: int,
    eval_batches: int = 20,
    checkpoint_interval: int | None = None,
    save_checkpoints: bool = False,
    save_best: bool = False,
) -> dict[str, Any]:
    return {
        "steps": steps,
        "batch_size": batch_size,
        "eval_interval": eval_interval,
        "eval_batches": eval_batches,
        "checkpoint_interval": checkpoint_interval or steps,
        "log_interval": min(10, eval_interval),
        "save_checkpoints": save_checkpoints,
        "save_best": save_best,
    }


def optimizer_override(learning_rate: float, *, warmup_iters: int, cycle_iters: int) -> dict[str, Any]:
    return {
        "max_lr": learning_rate,
        "min_lr": learning_rate * 0.1,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cycle_iters,
    }


def derived_config(
    base: str,
    run_name: str,
    *,
    optimizer: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    ablation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base": base,
        "run_name": run_name,
        "output_dir": f"runs/{run_name}",
    }
    if optimizer is not None:
        config["optimizer"] = optimizer
    if training is not None:
        config["training"] = training
    if ablation is not None:
        config["ablation"] = ablation
    return config


def build_configs(
    *,
    baseline_path: Path,
    output_dir: Path,
    selected_lr: float,
    selected_batch_size: int,
    selected_warmup_ratio: float,
) -> dict[str, dict[str, Any]]:
    baseline = os.path.relpath(baseline_path, output_dir)
    owt_baseline = os.path.relpath(Path("configs/owt_baseline.json").resolve(), output_dir)
    configs: dict[str, dict[str, Any]] = {}

    for learning_rate in LR_CANDIDATES:
        label = lr_label(learning_rate)
        short_name = f"lr_short_{label}"
        configs[f"{short_name}.json"] = derived_config(
            baseline,
            short_name,
            optimizer=optimizer_override(learning_rate, warmup_iters=50, cycle_iters=1_000),
            training=training_override(
                steps=1_000,
                batch_size=128,
                eval_interval=50,
                checkpoint_interval=1_000,
            ),
        )
        long_name = f"lr_long_{label}"
        configs[f"{long_name}.json"] = derived_config(
            baseline,
            long_name,
            optimizer=optimizer_override(learning_rate, warmup_iters=100, cycle_iters=5_000),
            training=training_override(
                steps=5_000,
                batch_size=128,
                eval_interval=100,
                checkpoint_interval=1_000,
            ),
        )

    for batch_size in THROUGHPUT_BATCHES:
        name = f"batch_throughput_{batch_size}"
        configs[f"{name}.json"] = derived_config(
            baseline,
            name,
            optimizer=optimizer_override(selected_lr, warmup_iters=5, cycle_iters=50),
            training=training_override(
                steps=50,
                batch_size=batch_size,
                eval_interval=50,
                eval_batches=2,
                checkpoint_interval=50,
            ),
        )

    for batch_size in QUALITY_BATCHES:
        tokens_per_step = batch_size * CONTEXT_LENGTH
        if QUALITY_TOKEN_BUDGET % tokens_per_step != 0:
            raise ValueError(f"token budget is not divisible by batch size {batch_size}")
        steps = QUALITY_TOKEN_BUDGET // tokens_per_step
        name = f"batch_equal_tokens_{batch_size}"
        configs[f"{name}.json"] = derived_config(
            baseline,
            name,
            optimizer=optimizer_override(
                selected_lr,
                warmup_iters=round(steps * 0.05),
                cycle_iters=steps,
            ),
            training=training_override(
                steps=steps,
                batch_size=batch_size,
                eval_interval=max(25, steps // 20),
                checkpoint_interval=steps,
            ),
        )

    warmup_steps = 1_000
    for warmup_ratio in WARMUP_RATIOS:
        warmup_iters = round(warmup_steps * warmup_ratio)
        ratio_label = f"{round(warmup_ratio * 100):02d}pct"
        name = f"warmup_{ratio_label}"
        configs[f"{name}.json"] = derived_config(
            baseline,
            name,
            optimizer=optimizer_override(
                selected_lr,
                warmup_iters=warmup_iters,
                cycle_iters=warmup_steps,
            ),
            training=training_override(
                steps=warmup_steps,
                batch_size=selected_batch_size,
                eval_interval=50,
                checkpoint_interval=warmup_steps,
            ),
        )

    final_steps = 10_000
    final_name = "tinystories_final"
    configs[f"{final_name}.json"] = derived_config(
        baseline,
        final_name,
        optimizer=optimizer_override(
            selected_lr,
            warmup_iters=round(final_steps * selected_warmup_ratio),
            cycle_iters=final_steps,
        ),
        training=training_override(
            steps=final_steps,
            batch_size=selected_batch_size,
            eval_interval=500,
            checkpoint_interval=1_000,
            save_checkpoints=True,
            save_best=True,
        ),
    )

    owt_final_name = "owt_final"
    configs[f"{owt_final_name}.json"] = derived_config(
        owt_baseline,
        owt_final_name,
        optimizer=optimizer_override(
            selected_lr,
            warmup_iters=round(final_steps * selected_warmup_ratio),
            cycle_iters=final_steps,
        ),
        training=training_override(
            steps=final_steps,
            batch_size=selected_batch_size,
            eval_interval=500,
            checkpoint_interval=1_000,
            save_checkpoints=True,
            save_best=True,
        ),
    )

    ablations = {
        "no_rmsnorm": {"remove_rmsnorm": True},
        "postnorm": {"use_post_norm": True},
        "nope": {"remove_rope": True},
        "silu": {"ffn_type": "silu"},
    }
    for label, ablation in ablations.items():
        name = f"tinystories_ablation_{label}"
        configs[f"{name}.json"] = derived_config(
            f"{final_name}.json",
            name,
            training={"save_checkpoints": False, "save_best": False},
            ablation=ablation,
        )

    no_rmsnorm_low_lr_name = "tinystories_ablation_no_rmsnorm_low_lr"
    configs[f"{no_rmsnorm_low_lr_name}.json"] = derived_config(
        f"{final_name}.json",
        no_rmsnorm_low_lr_name,
        optimizer=optimizer_override(
            selected_lr * 0.1,
            warmup_iters=round(final_steps * selected_warmup_ratio),
            cycle_iters=final_steps,
        ),
        training={"save_checkpoints": False, "save_best": False},
        ablation={"remove_rmsnorm": True},
    )

    return configs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = build_configs(
        baseline_path=Path("configs/tinystories_baseline.json").resolve(),
        output_dir=output_dir.resolve(),
        selected_lr=args.selected_lr,
        selected_batch_size=args.selected_batch_size,
        selected_warmup_ratio=args.selected_warmup_ratio,
    )
    for filename, config in configs.items():
        (output_dir / filename).write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "config_count": len(configs),
                "selected_lr": args.selected_lr,
                "selected_batch_size": args.selected_batch_size,
                "selected_warmup_ratio": args.selected_warmup_ratio,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
