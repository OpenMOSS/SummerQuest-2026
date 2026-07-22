#!/usr/bin/env python3
"""Prepare OWT after download, then train after TinyStories experiments finish."""

from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    Path("state").mkdir(exist_ok=True)
    failed = Path("state/owt_pipeline.failed")
    failed.unlink(missing_ok=True)
    try:
        raw_files = [Path("data/owt_train.txt"), Path("data/owt_valid.txt")]
        while not all(path.exists() for path in raw_files):
            print("waiting for OWT download and extraction", flush=True)
            time.sleep(60)
        subprocess.run([sys.executable, "scripts/prepare_data.py"], check=True)
        while not Path("state/tiny_main.done").exists():
            print("waiting for TinyStories GPU experiments", flush=True)
            time.sleep(60)
        subprocess.run([sys.executable, "scripts/run_suite.py", "--stage", "owt"], check=True)
        Path("state/owt_pipeline.done").touch()
    except Exception:
        failed.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
