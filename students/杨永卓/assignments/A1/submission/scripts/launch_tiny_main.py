#!/usr/bin/env python3
"""Run TinyStories baseline and ablations after the divergence probe finishes."""

from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    Path("state").mkdir(exist_ok=True)
    failed = Path("state/tiny_main.failed")
    failed.unlink(missing_ok=True)
    try:
        divergence_summary = Path("runs/lr_sweep/lr_divergent_3e-1/summary.json")
        while not divergence_summary.exists():
            print("waiting for divergence run", flush=True)
            time.sleep(30)
        subprocess.run([sys.executable, "scripts/run_suite.py", "--stage", "baseline"], check=True)
        subprocess.run([sys.executable, "scripts/run_suite.py", "--stage", "ablations"], check=True)
        Path("state/tiny_main.done").touch()
    except Exception:
        failed.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
