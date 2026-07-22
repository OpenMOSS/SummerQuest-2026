from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from typing import Any

import torch

from cs336_basics.Part5.configuration import ExperimentConfig, load_experiment_config
from cs336_basics.Part5.training import resolve_device, resolve_project_path, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the max-batch experiment from a batch probe result.")
    parser.add_argument("--probe", default="logs/batch_size/max_batch_probe.json")
    parser.add_argument("--config", default="configs/tinystories_batch_max.json")
    parser.add_argument("--target-tokens", type=int, default=4_194_304)
    parser.add_argument("--device", help="Override the configured device.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.target_tokens <= 0:
        raise ValueError("--target-tokens must be positive.")

    probe_value = json.loads(resolve_project_path(args.probe).read_text(encoding="utf-8"))
    if not isinstance(probe_value, dict):
        raise ValueError("batch probe result must be a JSON object.")

    config = load_experiment_config(resolve_project_path(args.config))
    device = resolve_device(args.device or config.training.device)
    batch_size = _validate_probe(probe_value, config=config, device=device)
    config = _build_run_config(config, batch_size=batch_size, target_tokens=args.target_tokens)
    summary = train_experiment(config, device_override=args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if summary["status"] == "diverged" else 0


def _build_run_config(
    config: ExperimentConfig,
    *,
    batch_size: int,
    target_tokens: int,
) -> ExperimentConfig:
    tokens_per_step = batch_size * config.model.context_length * config.training.gradient_accumulation_steps
    max_steps = math.ceil(target_tokens / tokens_per_step)
    warmup_steps = 0 if max_steps == 1 else max(1, max_steps // 16)
    eval_interval = max(1, math.ceil(max_steps / 16))
    log_interval = max(1, math.ceil(max_steps / 64))
    return replace(
        config,
        optimizer=replace(
            config.optimizer,
            warmup_steps=warmup_steps,
            cosine_cycle_steps=max_steps,
        ),
        training=replace(
            config.training,
            batch_size=batch_size,
            max_steps=max_steps,
            eval_interval=eval_interval,
            eval_batch_size=min(config.training.eval_batch_size, batch_size),
            log_interval=log_interval,
        ),
    )


def _validate_probe(
    probe_value: dict[str, Any],
    *,
    config: ExperimentConfig,
    device: torch.device,
) -> int:
    batch_size = probe_value.get("largest_successful_batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch probe result does not contain a positive largest_successful_batch_size.")
    first_failing_batch_size = probe_value.get("first_failing_batch_size")
    if probe_value.get("search_complete") is not True or (
        isinstance(first_failing_batch_size, bool)
        or not isinstance(first_failing_batch_size, int)
        or first_failing_batch_size <= batch_size
    ):
        raise ValueError(
            "batch probe did not observe a failing batch size; rerun it with a larger --upper-bound."
        )

    expected_model = config.to_dict()["model"]
    if probe_value.get("model") != expected_model:
        raise ValueError("batch probe model configuration does not match the experiment config.")
    if probe_value.get("dtype") != config.training.dtype:
        raise ValueError("batch probe dtype does not match the experiment config.")
    if probe_value.get("amp_dtype") != config.training.amp_dtype:
        raise ValueError("batch probe amp_dtype does not match the experiment config.")
    if probe_value.get("gradient_accumulation_steps") != config.training.gradient_accumulation_steps:
        raise ValueError("batch probe gradient accumulation does not match the experiment config.")

    probe_device_value = probe_value.get("device")
    if not isinstance(probe_device_value, str):
        raise ValueError("batch probe result does not contain a device string.")
    try:
        probe_device = torch.device(probe_device_value)
    except (RuntimeError, TypeError) as error:
        raise ValueError("batch probe result contains an invalid device.") from error
    if probe_device.type != device.type or (
        probe_device.index is not None and device.index is not None and probe_device.index != device.index
    ):
        raise ValueError("batch probe device does not match the requested training device.")
    return batch_size


if __name__ == "__main__":
    raise SystemExit(main())
