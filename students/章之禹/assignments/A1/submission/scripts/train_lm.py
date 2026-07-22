"""Train a Transformer language model from a TOML experiment config."""

from __future__ import annotations

import argparse
import json

from cs336_basics.training import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the TOML experiment configuration")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path, or 'latest' for the configured output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_from_config(args.config, resume=args.resume)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
