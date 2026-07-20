#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def write_config(output_dir: Path, name: str, config: dict) -> None:
    (output_dir / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create controlled TinyStories experiment configurations.")
    parser.add_argument("--baseline", default="configs/tinystories_baseline.json")
    parser.add_argument("--output-dir", default="artifacts/experiment_configs")
    parser.add_argument("--max-batch-size", type=int, default=256)
    args = parser.parse_args()

    with open(args.baseline, encoding="utf-8") as file:
        baseline = json.load(file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_config in output_dir.glob("*.json"):
        old_config.unlink()

    for learning_rate in [1e-4, 5e-4, 2e-3, 1e-2, 1.0]:
        config = copy.deepcopy(baseline)
        config.update(
            training_steps=500,
            warmup_steps=25,
            max_lr=learning_rate,
            min_lr=learning_rate / 10,
            val_interval=100,
            checkpoint_interval=500,
            val_batches=10,
        )
        label = f"{learning_rate:.0e}".replace("-0", "-").replace("e+00", "e0")
        write_config(output_dir, f"lr_{label}", config)

    for batch_size in sorted({1, 16, 64, 128, args.max_batch_size}):
        config = copy.deepcopy(baseline)
        config.update(
            batch_size=batch_size,
            training_steps=500,
            warmup_steps=25,
            val_interval=100,
            checkpoint_interval=500,
            val_batches=10,
        )
        write_config(output_dir, f"batch_{batch_size}", config)

    for ablation in ["baseline", "no_rmsnorm", "post_norm", "no_rope", "silu_ffn"]:
        config = copy.deepcopy(baseline)
        config.update(
            ablation=ablation,
            training_steps=1000,
            warmup_steps=50,
            val_interval=100,
            checkpoint_interval=1000,
            val_batches=10,
        )
        if ablation == "silu_ffn":
            config["silu_d_ff"] = 2048
        write_config(output_dir, f"ablation_{ablation}", config)

    no_norm_low_lr = copy.deepcopy(baseline)
    no_norm_low_lr.update(
        ablation="no_rmsnorm",
        max_lr=1e-4,
        min_lr=1e-5,
        training_steps=1000,
        warmup_steps=50,
        val_interval=100,
        checkpoint_interval=1000,
        val_batches=10,
    )
    write_config(output_dir, "ablation_no_rmsnorm_low_lr", no_norm_low_lr)


if __name__ == "__main__":
    main()
