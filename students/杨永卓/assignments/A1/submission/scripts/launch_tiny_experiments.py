#!/usr/bin/env python3
"""Wait for TinyStories token arrays, then run batch and LR probes."""

from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    Path("state").mkdir(exist_ok=True)
    failed = Path("state/tiny_initial.failed")
    failed.unlink(missing_ok=True)
    required = [Path("data/tinystories_train.bin"), Path("data/tinystories_valid.bin")]
    summaries = [
        Path("logs/tokenizers/tinystories_train_encode.json"),
        Path("logs/tokenizers/tinystories_valid_encode.json"),
    ]
    try:
        while not all(path.exists() for path in required + summaries):
            print("waiting for TinyStories encoding", flush=True)
            time.sleep(30)
        subprocess.run([sys.executable, "scripts/run_suite.py", "--stage", "batch"], check=True)
        subprocess.run([sys.executable, "scripts/run_suite.py", "--stage", "lr"], check=True)
        Path("state/tiny_initial.done").touch()
    except Exception:
        failed.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
