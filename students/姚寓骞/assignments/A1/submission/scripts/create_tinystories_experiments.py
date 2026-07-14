from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def save_config(base: dict, config_dir: Path, name: str, changes: dict) -> None:
    config = copy.deepcopy(base)
    config["output_dir"] = f"artifacts/runs/{name}"
    for key, value in changes.items():
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    path = config_dir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated TinyStories experiment configs")
    parser.add_argument("--base", type=Path, default=Path("configs/tinystories_baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--lr-steps", type=int, default=10000)
    parser.add_argument("--batch-steps", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 128, 256, 512, 1024])
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    save_config(base, args.output, "tinystories_baseline", {})

    for learning_rate in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        label = f"{learning_rate:.0e}".replace("-0", "-")
        save_config(
            base,
            args.output,
            f"lr_sweep/lr_{label}",
            {
                "optimizer.lr": learning_rate,
                "max_lr": learning_rate,
                "min_lr": learning_rate / 10,
                "max_steps": args.lr_steps,
                "warmup_steps": min(100, args.lr_steps // 10),
                "checkpoint_every": args.lr_steps,
            },
        )

    for batch_size in args.batch_sizes:
        save_config(
            base,
            args.output,
            f"batch_size/batch_{batch_size}",
            {
                "batch_size": batch_size,
                "max_steps": args.batch_steps,
                "warmup_steps": min(10, args.batch_steps // 10),
                "eval_every": args.batch_steps,
                "checkpoint_every": args.batch_steps,
            },
        )

    ablations = {
        "ablation_no_rmsnorm": {"model.use_rmsnorm": False},
        "ablation_postnorm": {"model.norm_position": "post"},
        "ablation_nope": {"model.position_encoding": "none"},
        # 2 * 512 * 2016 approximately matches 3 * 512 * 1344 parameters.
        "ablation_silu": {"model.ffn_type": "silu", "model.d_ff": 2016},
    }
    for name, changes in ablations.items():
        save_config(base, args.output, name, changes)


if __name__ == "__main__":
    main()
