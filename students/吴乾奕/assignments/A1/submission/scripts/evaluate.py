#!/usr/bin/env python3
"""Evaluate a checkpoint on random memory-mapped validation batches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from cs336_basics.config import apply_overrides, load_json_config, project_root, resolve_project_path
from cs336_basics.experiment import (
    build_transformer,
    canonical_model_config,
    checkpoint_model_config,
    extract_model_state,
    load_checkpoint_payload,
    parameter_count,
)
from cs336_basics.training import estimate_loss, load_token_array, resolve_device, resolve_dtype, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--allow-data-override", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def equivalent_data_config(left: dict[str, object], right: dict[str, object], root: Path) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        if key.endswith("_tokens") or key.endswith("_dir"):
            left_path = resolve_project_path(left[key], root=root) if left[key] is not None else None
            right_path = resolve_project_path(right[key], root=root) if right[key] is not None else None
            if left_path != right_path:
                return False
        elif left[key] != right[key]:
            return False
    return True


def main() -> None:
    args = parse_args()
    root = project_root()
    config = apply_overrides(load_json_config(args.config), args.overrides)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    compute_dtype = resolve_dtype(str(config.get("dtype", "float32")), device)
    parameter_dtype = resolve_dtype(str(config.get("parameter_dtype", "float32")), device)
    amp_enabled = bool(config.get("amp", device.type in {"cuda", "mps"})) and compute_dtype != torch.float32

    checkpoint_path = resolve_project_path(args.checkpoint, root=root)
    assert checkpoint_path is not None
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    model_config = checkpoint_model_config(payload, config["model"])
    if canonical_model_config(model_config) != canonical_model_config(config["model"]):
        print(
            "warning: using architecture stored in the checkpoint instead of the external config",
            file=sys.stderr,
        )
    model = build_transformer(model_config, device=device, dtype=parameter_dtype)
    model.load_state_dict(extract_model_state(payload))

    checkpoint_config = payload.get("config", {})
    checkpoint_data = checkpoint_config.get("data") if isinstance(checkpoint_config, dict) else None
    data_config = config["data"]
    if isinstance(checkpoint_data, dict) and not equivalent_data_config(checkpoint_data, data_config, root):
        if not args.allow_data_override:
            raise ValueError(
                "external data configuration differs from the checkpoint; "
                "pass --allow-data-override to evaluate intentionally on another dataset"
            )
        print("warning: evaluating on the externally supplied data configuration", file=sys.stderr)
    else:
        data_config = checkpoint_data or data_config
    validation_path = resolve_project_path(data_config["validation_tokens"], root=root)
    assert validation_path is not None
    validation_data = load_token_array(validation_path)
    external_training = config["training"]
    checkpoint_training = checkpoint_config.get("training") if isinstance(checkpoint_config, dict) else None
    default_training = checkpoint_training if isinstance(checkpoint_training, dict) else external_training
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(default_training.get("validation_batch_size", default_training["batch_size"]))
    )
    num_batches = (
        args.num_batches if args.num_batches is not None else int(default_training.get("validation_batches", 20))
    )
    if batch_size <= 0 or num_batches <= 0:
        raise ValueError("batch size and number of validation batches must be positive")
    rng = np.random.default_rng(seed + 10_000)

    validation_loss = estimate_loss(
        model,
        validation_data,
        batch_size=batch_size,
        context_length=int(model_config["context_length"]),
        num_batches=num_batches,
        device=device,
        rng=rng,
        amp_dtype=compute_dtype,
        amp_enabled=amp_enabled,
    )
    if not math.isfinite(validation_loss):
        raise FloatingPointError(f"non-finite validation loss: {validation_loss}")
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": int(payload.get("iteration", -1)),
        "validation_loss": validation_loss,
        "perplexity": math.exp(validation_loss) if validation_loss < 80 else None,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "parameter_count": parameter_count(model),
        "device": str(device),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output_path = resolve_project_path(args.output, root=root)
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
