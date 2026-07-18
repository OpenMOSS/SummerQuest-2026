from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    ("tinystories_small", "configs/tinystories_small.json"),
    ("lr_sweep_3e-4", "configs/lr_sweep_3e-4.json"),
    ("lr_sweep_3e-3", "configs/lr_sweep_3e-3.json"),
    ("lr_sweep_1e1_diverge", "configs/lr_sweep_1e1_diverge.json"),
    ("batch_size_1", "configs/batch_size_1.json"),
    ("batch_size_32", "configs/batch_size_32.json"),
    ("batch_size_64", "configs/batch_size_64.json"),
    ("batch_size_256", "configs/batch_size_256.json"),
    ("ablation_no_rmsnorm", "configs/ablation_no_rmsnorm.json"),
    ("ablation_post_norm", "configs/ablation_post_norm.json"),
    ("ablation_nope", "configs/ablation_nope.json"),
    ("ablation_silu_ffn", "configs/ablation_silu_ffn.json"),
]

OWT_EXPERIMENT = ("owt_small", "configs/owt_small.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-owt", action="store_true")
    parser.add_argument("--start-from", choices=[name for name, _ in EXPERIMENTS + [OWT_EXPERIMENT]])
    parser.add_argument("--only", nargs="+", choices=[name for name, _ in EXPERIMENTS + [OWT_EXPERIMENT]])
    return parser.parse_args()


def load_config(repo_root: Path, config_path: str) -> dict:
    with open(repo_root / config_path, encoding="utf-8") as file:
        return json.load(file)


def check_inputs(repo_root: Path, name: str, config_path: str) -> dict:
    config = load_config(repo_root, config_path)
    required_paths = [
        config["train_tokens"],
        config["valid_tokens"],
        config["tokenizer"],
    ]
    missing = [path for path in required_paths if not (repo_root / path).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"{name} is missing required files: {joined}")
    return config


def should_skip(repo_root: Path, config: dict) -> bool:
    output_dir = repo_root / config["output_dir"]
    return (output_dir / config.get("summary_name", "summary.json")).exists()


def selected_experiments(args: argparse.Namespace) -> list[tuple[str, str]]:
    experiments = list(EXPERIMENTS)
    wants_owt = args.include_owt or args.start_from == OWT_EXPERIMENT[0] or (
        args.only is not None and OWT_EXPERIMENT[0] in args.only
    )
    if wants_owt:
        experiments.append(OWT_EXPERIMENT)
    if args.only:
        wanted = set(args.only)
        experiments = [item for item in experiments if item[0] in wanted]
    if args.start_from:
        names = [name for name, _ in experiments]
        start = names.index(args.start_from)
        experiments = experiments[start:]
    return experiments


def run_experiment(repo_root: Path, name: str, config_path: str) -> int:
    log_dir = repo_root / "runs" / "experiment_driver"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.out"
    cmd = [sys.executable, "scripts/train_lm.py", "--config", config_path]
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start}] START {name}: {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
    end = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{end}] END {name}: exit_code={process.returncode}, driver_log={log_path}", flush=True)
    return int(process.returncode)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    experiments = selected_experiments(args)
    if not experiments:
        raise SystemExit("No experiments selected.")

    print("Experiment order:")
    checked: list[tuple[str, str, dict]] = []
    for index, (name, config_path) in enumerate(experiments, start=1):
        config = check_inputs(repo_root, name, config_path)
        status = "skip-existing" if args.skip_existing and should_skip(repo_root, config) else "run"
        print(f"{index:02d}. {name}: {config_path} [{status}]")
        checked.append((name, config_path, config))

    if args.dry_run:
        return

    overall_start = time.time()
    for name, config_path, config in checked:
        if args.skip_existing and should_skip(repo_root, config):
            print(f"SKIP {name}: summary already exists")
            continue
        exit_code = run_experiment(repo_root, name, config_path)
        if exit_code != 0:
            raise SystemExit(exit_code)

    print(f"All selected experiments finished in {time.time() - overall_start:.1f} seconds.")


if __name__ == "__main__":
    main()
