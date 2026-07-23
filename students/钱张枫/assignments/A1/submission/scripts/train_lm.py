from __future__ import annotations

import argparse
import json

from cs336_basics.Part5.configuration import load_experiment_config
from cs336_basics.Part5.training import resolve_project_path, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a configurable CS336 Transformer language model.")
    parser.add_argument("--config", required=True, help="Path to an experiment JSON config.")
    parser.add_argument("--device", help="Override the configured device, for example cpu, mps, or cuda:0.")
    parser.add_argument("--resume", help="Override the checkpoint path used to resume training.")
    parser.add_argument("--max-steps", type=int, help="Override the configured maximum number of optimizer steps.")
    parser.add_argument("--batch-size", type=int, help="Override the configured micro batch size.")
    parser.add_argument("--max-learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--cosine-cycle-steps", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--log-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve and validate the config without loading data or training.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_experiment_config(resolve_project_path(args.config))
    if args.validate_config_only:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    summary = train_experiment(
        config,
        device_override=args.device,
        resume_override=args.resume,
        max_steps_override=args.max_steps,
        batch_size_override=args.batch_size,
        max_learning_rate_override=args.max_learning_rate,
        min_learning_rate_override=args.min_learning_rate,
        warmup_steps_override=args.warmup_steps,
        cosine_cycle_steps_override=args.cosine_cycle_steps,
        experiment_name_override=args.experiment_name,
        log_dir_override=args.log_dir,
        checkpoint_dir_override=args.checkpoint_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if summary["status"] == "diverged" else 0


if __name__ == "__main__":
    raise SystemExit(main())
