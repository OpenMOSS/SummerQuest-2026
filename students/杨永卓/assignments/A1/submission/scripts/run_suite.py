#!/usr/bin/env python3
"""Resumable driver for batch, LR, baseline, ablation, and OWT experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path


STAGES = {
    "lr": ["lr_1e-4.json", "lr_3e-4.json", "lr_1e-3.json", "lr_divergent.json"],
    "baseline": ["tinystories_baseline.json"],
    "ablations": [
        "ablation_control.json",
        "ablation_no_rmsnorm.json",
        "ablation_post_norm.json",
        "ablation_no_rope.json",
        "ablation_silu.json",
    ],
    "owt": ["owt_baseline.json"],
}


def run_training(config_path: Path) -> None:
    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    if (output_dir / "summary.json").exists():
        print(f"skip complete: {config['run_name']}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "driver.log"
    print(f"run: {config['run_name']}", flush=True)
    with log_path.open("a", encoding="utf-8") as output:
        subprocess.run(
            [sys.executable, "scripts/train_lm.py", "--config", str(config_path), "--resume"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["batch", *STAGES, "all"], required=True)
    args = parser.parse_args()
    Path("state").mkdir(exist_ok=True)
    failed = Path(f"state/{args.stage}.failed")
    failed.unlink(missing_ok=True)
    try:
        selected = ["batch", *STAGES] if args.stage == "all" else [args.stage]
        for stage in selected:
            if stage == "batch":
                output = Path("runs/batch_size/summary.json")
                if not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/batch_size_sweep.py",
                            "--config",
                            "configs/tinystories_baseline.json",
                            "--output",
                            str(output),
                        ],
                        check=True,
                    )
                continue
            for filename in STAGES[stage]:
                run_training(Path("configs") / filename)
        Path(f"state/{args.stage}.done").touch()
    except Exception:
        failed.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
