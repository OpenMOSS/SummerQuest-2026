"""Command-line entry point for Transformer LM training.

Example:
    uv run python scripts/train_lm.py --config configs/tinystories.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics.training import TrainingConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON file containing TrainingConfig fields")
    parser.add_argument("--device", help="override the device from the JSON configuration")
    parser.add_argument("--resume-from", help="override the checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device is not None:
        values["device"] = args.device
    if args.resume_from is not None:
        values["resume_from"] = args.resume_from
    train(TrainingConfig(**values))


if __name__ == "__main__":
    main()
