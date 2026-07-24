"""Run a battery of ablation studies on TinyStories."""

import argparse
import json
import subprocess
from pathlib import Path


ABLATIONS = {
    "baseline": {},
    "no_rmsnorm": {"--no_rmsnorm": True},
    "post_norm": {"--use_post_norm": True},
    "no_rope": {"--no_rope": True},
    "silu_ffn": {"--ffn_type": "silu"},
}


def build_command(
    base_args: list[str],
    output_dir: Path,
    name: str,
    overrides: dict,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "cs336_basics/train.py",
        *base_args,
        "--output_dir",
        str(output_dir / name),
        "--log_file",
        str(output_dir / name / "train.log"),
    ]
    for flag, value in overrides.items():
        if isinstance(value, bool) and value:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run A1 ablation experiments.")
    parser.add_argument("--train_tokens", type=str, required=True)
    parser.add_argument("--val_tokens", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ablations")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5_000)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=2_500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--train_tokens",
        args.train_tokens,
        "--val_tokens",
        args.val_tokens,
        "--vocab_path",
        args.vocab_path,
        "--merges_path",
        args.merges_path,
        "--vocab_size",
        str(args.vocab_size),
        "--context_length",
        str(args.context_length),
        "--d_model",
        str(args.d_model),
        "--num_layers",
        str(args.num_layers),
        "--num_heads",
        str(args.num_heads),
        "--d_ff",
        str(args.d_ff),
        "--batch_size",
        str(args.batch_size),
        "--max_iters",
        str(args.max_iters),
        "--learning_rate",
        str(args.learning_rate),
        "--eval_interval",
        str(args.eval_interval),
        "--checkpoint_interval",
        str(args.checkpoint_interval),
        "--device",
        args.device,
    ]

    plan = {}
    for name, overrides in ABLATIONS.items():
        ablation_overrides = dict(overrides)
        if name == "silu_ffn":
            # Match SwiGLU parameter count: d_ff = 4 * d_model.
            ablation_overrides["--d_ff"] = 4 * args.d_model
        cmd = build_command(base_args, output_dir, name, ablation_overrides)
        plan[name] = cmd

    with open(output_dir / "plan.json", "w") as f:
        json.dump({k: " ".join(v) for k, v in plan.items()}, f, indent=2)

    for name, cmd in plan.items():
        print(f"\n=== Running ablation: {name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"WARNING: ablation {name} exited with code {result.returncode}")

    print(f"\nAll ablations finished. Results are in {output_dir}")


if __name__ == "__main__":
    main()
