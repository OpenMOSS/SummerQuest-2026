#!/usr/bin/env python3
"""Generate samples and export public artifacts after all training completes."""

from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path


def run(command: list[str]) -> None:
    print("run:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    Path("state").mkdir(exist_ok=True)
    Path("runs/generation").mkdir(parents=True, exist_ok=True)
    failed = Path("state/generation_export.failed")
    failed.unlink(missing_ok=True)
    try:
        while not Path("state/owt_pipeline.done").exists():
            print("waiting for all training", flush=True)
            time.sleep(60)
        run(
            [
                sys.executable,
                "scripts/generate.py",
                "--config",
                "configs/tinystories_baseline.json",
                "--checkpoint",
                "runs/tinystories_baseline/checkpoint_last.pt",
                "--tokenizer",
                "data/tinystories_tokenizer.json",
                "--prompt",
                "Once upon a time",
                "--max-new-tokens",
                "256",
                "--temperature",
                "0.8",
                "--top-p",
                "0.95",
                "--output",
                "runs/generation/tinystories.txt",
            ]
        )
        run(
            [
                sys.executable,
                "scripts/generate.py",
                "--config",
                "configs/owt_baseline.json",
                "--checkpoint",
                "runs/owt_baseline/checkpoint_last.pt",
                "--tokenizer",
                "data/owt_tokenizer.json",
                "--prompt",
                "Language models are",
                "--max-new-tokens",
                "256",
                "--temperature",
                "0.8",
                "--top-p",
                "0.95",
                "--output",
                "runs/generation/owt.txt",
            ]
        )
        run([sys.executable, "scripts/export_results.py"])
        Path("state/generation_export.done").touch()
    except Exception:
        failed.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
