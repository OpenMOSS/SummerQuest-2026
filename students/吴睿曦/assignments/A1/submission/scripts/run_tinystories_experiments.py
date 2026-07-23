import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_experiment(config, train_data, val_data, output_root, name, overrides, keep_checkpoints):
    output_dir = output_root / name
    summary = output_dir / "summary.json"
    if summary.exists():
        print(f"skip completed run: {name}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        str(config),
        "--device",
        "cuda",
        "--set",
        f"train_data={json.dumps(str(train_data))}",
        "--set",
        f"val_data={json.dumps(str(val_data))}",
        "--set",
        f"output_dir={json.dumps(str(output_dir))}",
    ]
    for key, value in overrides.items():
        command.extend(("--set", f"{key}={json.dumps(value)}"))
    print(f"start run: {name}", flush=True)
    with open(output_dir / "console.log", "w", encoding="utf-8") as console:
        subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, check=True)
    if not keep_checkpoints:
        completed_steps = json.loads(summary.read_text(encoding="utf-8"))["completed_steps"]
        final_checkpoint = output_dir / f"checkpoint_{completed_steps}.pt"
        for checkpoint in output_dir.glob("checkpoint_*.pt"):
            if checkpoint != final_checkpoint:
                checkpoint.unlink()
    print(f"completed run: {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the required TinyStories comparison suite")
    parser.add_argument("config", type=Path)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, required=True, help="Largest batch that passed a capacity probe")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    experiments = [
        ("lr_1e-4", {"max_lr": 1e-4, "min_lr": 1e-5}),
        ("lr_6e-4", {"max_lr": 6e-4, "min_lr": 6e-5}),
        ("lr_1e-3", {"max_lr": 1e-3, "min_lr": 1e-4}),
        ("lr_1e-1_divergent", {"max_lr": 1e-1, "min_lr": 1e-2}),
        (
            "batch_1",
            {"batch_size": 1, "max_iters": 100, "warmup_iters": 10, "eval_interval": 25, "eval_batches": 5,
             "checkpoint_interval": 100},
        ),
        (
            "batch_64",
            {"batch_size": 64, "max_iters": 100, "warmup_iters": 10, "eval_interval": 25,
             "eval_batches": 5, "checkpoint_interval": 100},
        ),
        (
            "batch_128",
            {"batch_size": 128, "max_iters": 100, "warmup_iters": 10, "eval_interval": 25,
             "eval_batches": 5, "checkpoint_interval": 100},
        ),
        (
            f"batch_{args.max_batch}",
            {"batch_size": args.max_batch, "max_iters": 100, "warmup_iters": 10, "eval_interval": 25,
             "eval_batches": 5, "checkpoint_interval": 100},
        ),
        ("ablation_no_rmsnorm", {"norm_style": "none"}),
        ("ablation_post_norm", {"norm_style": "post"}),
        ("ablation_nope", {"use_rope": False}),
        ("ablation_silu", {"ffn_variant": "silu", "d_ff": 2016}),
    ]
    for name, overrides in experiments:
        run_experiment(
            args.config,
            args.train_data,
            args.val_data,
            args.output_root,
            name,
            overrides,
            args.keep_checkpoints,
        )


if __name__ == "__main__":
    main()
