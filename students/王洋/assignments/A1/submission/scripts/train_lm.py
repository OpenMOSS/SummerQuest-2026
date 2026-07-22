#!/usr/bin/env python3
"""Config-driven Transformer LM training with mmap data and JSONL metrics."""

from __future__ import annotations

import argparse
import copy
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import AdamW, cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping

from _common import atomic_write_json, load_json, resolve_device, synchronize_device, utc_timestamp


DEFAULTS: dict[str, Any] = {
    "seed": 1337,
    "device": "auto",
    "precision": "float32",
    "deterministic": False,
    "allow_tf32": True,
    "optimizer": {
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    },
    "schedule": {
        "min_learning_rate": 3e-5,
        "warmup_iters": 100,
        "cosine_cycle_iters": None,
    },
    "training": {
        "max_steps": 10_000,
        "micro_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
    },
    "evaluation": {
        "interval": 100,
        "batches": 20,
        "batch_size": 16,
        "at_start": True,
    },
    "logging": {"interval": 10},
    "checkpoint": {"interval": 500},
}

NONFINITE_FIELDS_KEY = "nonfinite_fields"


def _normalize_json_value(value: Any, path: str, nonfinite_fields: dict[str, str]) -> Any:
    """Return a strict-JSON value, recording every replaced NaN or infinity."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            marker = "NaN"
        elif value > 0:
            marker = "+Infinity"
        else:
            marker = "-Infinity"
        nonfinite_fields[path] = marker
        return None
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path or '<root>'} must be a string")
            child_path = f"{path}.{key}" if path else key
            normalized[key] = _normalize_json_value(item, child_path, nonfinite_fields)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item, f"{path}[{index}]", nonfinite_fields) for index, item in enumerate(value)]
    return value


def strict_json_dumps(payload: dict[str, Any]) -> str:
    """Serialize one event as standards-compliant JSON with auditable non-finite markers."""

    existing_markers = payload.get(NONFINITE_FIELDS_KEY)
    if existing_markers is not None and (
        not isinstance(existing_markers, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in existing_markers.items())
    ):
        raise TypeError(f"{NONFINITE_FIELDS_KEY} must be a string-to-string object")

    nonfinite_fields: dict[str, str] = dict(existing_markers or {})
    body = {key: value for key, value in payload.items() if key != NONFINITE_FIELDS_KEY}
    normalized = _normalize_json_value(body, "", nonfinite_fields)
    if nonfinite_fields:
        normalized[NONFINITE_FIELDS_KEY] = dict(sorted(nonfinite_fields.items()))
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer from a JSON configuration.")
    parser.add_argument("--config", type=Path, required=True, help="JSON experiment configuration")
    parser.add_argument("--train-data", type=Path, help="override data.train")
    parser.add_argument("--validation-data", type=Path, help="override data.validation")
    parser.add_argument("--output-dir", type=Path, help="override output_dir")
    parser.add_argument("--device", help="override device (auto, cpu, cuda, cuda:N, or mps)")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), help="override autocast precision")
    parser.add_argument("--seed", type=int, help="override seed")
    parser.add_argument("--max-steps", type=int, help="override training.max_steps")
    parser.add_argument("--learning-rate", type=float, help="override optimizer.learning_rate")
    parser.add_argument("--min-learning-rate", type=float, help="override schedule.min_learning_rate")
    parser.add_argument("--warmup-iters", type=int, help="override schedule.warmup_iters")
    parser.add_argument("--cosine-cycle-iters", type=int, help="override schedule.cosine_cycle_iters")
    parser.add_argument("--micro-batch-size", type=int, help="override training.micro_batch_size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="override training.gradient_accumulation_steps",
    )
    parser.add_argument("--evaluation-interval", type=int, help="override evaluation.interval")
    parser.add_argument("--evaluation-batches", type=int, help="override evaluation.batches")
    parser.add_argument("--evaluation-batch-size", type=int, help="override evaluation.batch_size")
    parser.add_argument(
        "--evaluation-at-start",
        action=argparse.BooleanOptionalAction,
        help="override evaluation.at_start",
    )
    parser.add_argument("--logging-interval", type=int, help="override logging.interval")
    parser.add_argument("--checkpoint-interval", type=int, help="override checkpoint.interval")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="resume a checkpoint; omit the value to use OUTPUT_DIR/latest.pt",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace managed run files for a fresh run")
    return parser.parse_args()


def merge_defaults(value: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, default in defaults.items():
        current = value.get(key, default)
        if isinstance(default, dict):
            if not isinstance(current, dict):
                raise ValueError(f"configuration field {key!r} must be an object")
            merged[key] = merge_defaults(current, default)
        else:
            merged[key] = current
    for key, current in value.items():
        if key not in merged:
            merged[key] = current
    return merged


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.train_data is not None:
        config.setdefault("data", {})["train"] = str(args.train_data)
    if args.validation_data is not None:
        config.setdefault("data", {})["validation"] = str(args.validation_data)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.device is not None:
        config["device"] = args.device
    if args.precision is not None:
        config["precision"] = args.precision
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        config["optimizer"]["learning_rate"] = args.learning_rate
    if args.min_learning_rate is not None:
        config["schedule"]["min_learning_rate"] = args.min_learning_rate
    if args.warmup_iters is not None:
        config["schedule"]["warmup_iters"] = args.warmup_iters
    if args.cosine_cycle_iters is not None:
        config["schedule"]["cosine_cycle_iters"] = args.cosine_cycle_iters
    if args.micro_batch_size is not None:
        config["training"]["micro_batch_size"] = args.micro_batch_size
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.evaluation_interval is not None:
        config["evaluation"]["interval"] = args.evaluation_interval
    if args.evaluation_batches is not None:
        config["evaluation"]["batches"] = args.evaluation_batches
    if args.evaluation_batch_size is not None:
        config["evaluation"]["batch_size"] = args.evaluation_batch_size
    if args.evaluation_at_start is not None:
        config["evaluation"]["at_start"] = args.evaluation_at_start
    if args.logging_interval is not None:
        config["logging"]["interval"] = args.logging_interval
    if args.checkpoint_interval is not None:
        config["checkpoint"]["interval"] = args.checkpoint_interval


def require_positive(config: dict[str, Any], section: str, key: str, *, allow_zero: bool = False) -> int:
    value = config[section][key]
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{section}.{key} must be a {qualifier} integer")
    return value


def _nonfinite_number_paths(value: Any, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(path or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            paths.extend(_nonfinite_number_paths(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_nonfinite_number_paths(item, f"{path}[{index}]"))
    return paths


def validate_config(config: dict[str, Any]) -> None:
    nonfinite_paths = _nonfinite_number_paths(config)
    if nonfinite_paths:
        raise ValueError(f"configuration contains non-finite numeric values at: {', '.join(nonfinite_paths)}")
    for required in ("model", "data", "output_dir"):
        if required not in config:
            raise ValueError(f"configuration is missing required field {required!r}")
    if not isinstance(config["model"], dict) or not isinstance(config["data"], dict):
        raise ValueError("model and data must be JSON objects")
    if "train" not in config["data"]:
        raise ValueError("configuration is missing data.train")
    if config["precision"] not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")

    require_positive(config, "training", "max_steps")
    require_positive(config, "training", "micro_batch_size")
    require_positive(config, "training", "gradient_accumulation_steps")
    require_positive(config, "evaluation", "interval")
    require_positive(config, "evaluation", "batches")
    require_positive(config, "evaluation", "batch_size")
    require_positive(config, "logging", "interval")
    require_positive(config, "checkpoint", "interval")
    if float(config["training"]["gradient_clip_norm"]) <= 0:
        raise ValueError("training.gradient_clip_norm must be positive")
    maximum_learning_rate = float(config["optimizer"]["learning_rate"])
    minimum_learning_rate = float(config["schedule"]["min_learning_rate"])
    if maximum_learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be positive")
    if minimum_learning_rate < 0 or minimum_learning_rate > maximum_learning_rate:
        raise ValueError("schedule.min_learning_rate must be between zero and optimizer.learning_rate")
    warmup_iters = config["schedule"]["warmup_iters"]
    cosine_cycle_iters = config["schedule"]["cosine_cycle_iters"]
    if not isinstance(warmup_iters, int) or isinstance(warmup_iters, bool) or warmup_iters < 0:
        raise ValueError("schedule.warmup_iters must be a non-negative integer")
    if cosine_cycle_iters is not None and (
        not isinstance(cosine_cycle_iters, int)
        or isinstance(cosine_cycle_iters, bool)
        or cosine_cycle_iters < warmup_iters
    ):
        raise ValueError("schedule.cosine_cycle_iters must be null or an integer at least warmup_iters")
    if cosine_cycle_iters is None and config["training"]["max_steps"] < warmup_iters:
        raise ValueError("training.max_steps must be at least warmup_iters when cosine_cycle_iters is null")

    model = config["model"]
    required_model_fields = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff")
    missing = [key for key in required_model_fields if key not in model]
    if missing:
        raise ValueError(f"model is missing required fields: {', '.join(missing)}")


def load_token_array(path: str | os.PathLike[str], context_length: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{path} must contain a one-dimensional integer array")
    if len(array) <= context_length:
        raise ValueError(f"{path} must contain more than {context_length} tokens")
    return array


def load_token_array_provenance(
    path: str | os.PathLike[str],
    array: np.ndarray,
    *,
    expected_vocab_size: int,
) -> dict[str, Any]:
    """Validate an encoder sidecar and return stable checkpoint provenance."""

    array_path = Path(path)
    sidecar_path = array_path.with_name(f"{array_path.name}.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"missing token-array sidecar {sidecar_path}; encode data with scripts/encode_dataset.py"
        )
    metadata = load_json(sidecar_path)
    if metadata.get("format") != "cs336-token-array-v1":
        raise ValueError(f"{sidecar_path} has unsupported format {metadata.get('format')!r}")

    expected_shape = list(array.shape)
    if metadata.get("shape") != expected_shape:
        raise ValueError(
            f"{sidecar_path} shape {metadata.get('shape')!r} does not match array shape {expected_shape!r}"
        )
    token_count = metadata.get("token_count")
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count != len(array):
        raise ValueError(f"{sidecar_path} token_count does not match the array length")
    dtype_name = array.dtype.name
    if metadata.get("dtype") != dtype_name:
        raise ValueError(f"{sidecar_path} dtype {metadata.get('dtype')!r} does not match array dtype {dtype_name!r}")
    tokenizer_vocab_size = metadata.get("tokenizer_vocab_size")
    if (
        not isinstance(tokenizer_vocab_size, int)
        or isinstance(tokenizer_vocab_size, bool)
        or tokenizer_vocab_size != expected_vocab_size
    ):
        raise ValueError(
            f"{sidecar_path} tokenizer_vocab_size {tokenizer_vocab_size!r} "
            f"does not match model vocab_size {expected_vocab_size}"
        )

    tokenizer_sha256 = metadata.get("tokenizer_sha256")
    required_hashes = ("vocab_file", "merges_file")
    if not isinstance(tokenizer_sha256, dict):
        raise ValueError(f"{sidecar_path} is missing tokenizer_sha256 provenance")
    normalized_hashes: dict[str, str] = {}
    for key in required_hashes:
        digest = tokenizer_sha256.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError(f"{sidecar_path} has an invalid tokenizer_sha256.{key}")
        normalized_hashes[key] = digest.lower()

    return {
        "sidecar_filename": sidecar_path.name,
        "format": metadata["format"],
        "shape": expected_shape,
        "token_count": token_count,
        "dtype": dtype_name,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tokenizer_sha256": normalized_hashes,
    }


def attach_data_provenance(
    config: dict[str, Any],
    train_data: np.ndarray,
    validation_data: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    vocab_size = int(config["model"]["vocab_size"])
    provenance = {
        "train": load_token_array_provenance(
            config["data"]["train"],
            train_data,
            expected_vocab_size=vocab_size,
        )
    }
    if validation_data is not None:
        provenance["validation"] = load_token_array_provenance(
            config["data"]["validation"],
            validation_data,
            expected_vocab_size=vocab_size,
        )
        if provenance["validation"]["tokenizer_sha256"] != provenance["train"]["tokenizer_sha256"]:
            raise ValueError("training and validation arrays were encoded by different tokenizers")

    declared = config.get("data_provenance")
    if declared is not None and declared != provenance:
        raise ValueError("configured data_provenance does not match the observed token-array sidecars")
    config["data_provenance"] = provenance
    return provenance


def precision_context(device: torch.device, precision: str) -> contextlib.AbstractContextManager:
    if precision == "float32":
        return contextlib.nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"bfloat16 autocast is not supported by this script on {device.type}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    *,
    batches: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
    precision: str,
    evaluation_seed: int,
) -> float:
    was_training = model.training
    numpy_state = np.random.get_state()
    np.random.seed(evaluation_seed % (2**32))
    losses: list[float] = []
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(batches):
                inputs, targets = get_batch(dataset, batch_size, context_length, device)
                with precision_context(device, precision):
                    loss = cross_entropy(model(inputs), targets)
                losses.append(float(loss.float().item()))
    finally:
        np.random.set_state(numpy_state)
        model.train(was_training)
    return sum(losses) / len(losses)


def gradient_norm(parameters: torch.nn.Module) -> float:
    squared_norm = torch.zeros((), device=next(parameters.parameters()).device, dtype=torch.float32)
    for parameter in parameters.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().square().sum()
    return float(torch.sqrt(squared_norm).item())


def device_memory_metrics(device: torch.device) -> dict[str, int]:
    """Return CUDA allocator statistics suitable for JSONL logging."""

    if device.type != "cuda":
        return {}
    return {
        "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
        "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_training_checkpoint(
    path: Path,
    *,
    model: TransformerLM,
    optimizer: AdamW,
    step: int,
    processed_tokens: int,
    wall_time_seconds: float,
    config: dict[str, Any],
    status: str,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    checkpoint = {
        "format": "cs336-training-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": step,
        "processed_tokens": processed_tokens,
        "wall_time_seconds": wall_time_seconds,
        "config": config,
        "rng_state": capture_rng_state(),
        "status": status,
    }
    if diagnostics is not None:
        checkpoint["diagnostics"] = diagnostics
    try:
        with temporary.open("wb") as output_file:
            torch.save(checkpoint, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resume_immutable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Project a resolved config onto fields that define optimizer trajectory."""

    immutable = copy.deepcopy(config)
    for key in ("output_dir", "run_name", "device"):
        immutable.pop(key, None)

    training = immutable.get("training")
    if isinstance(training, dict):
        training.pop("max_steps", None)
    evaluation = immutable.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation.pop("interval", None)
        evaluation.pop("at_start", None)
    logging = immutable.get("logging")
    if isinstance(logging, dict):
        logging.pop("interval", None)
    checkpoint = immutable.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("interval", None)
    return immutable


def _config_differences(saved: Any, requested: Any, path: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(requested, dict):
        differences: list[str] = []
        for key in sorted(set(saved) | set(requested)):
            child_path = f"{path}.{key}" if path else key
            if key not in saved:
                differences.append(f"{child_path} (missing from checkpoint)")
            elif key not in requested:
                differences.append(f"{child_path} (missing from requested config)")
            else:
                differences.extend(_config_differences(saved[key], requested[key], child_path))
        return differences
    if saved != requested:
        return [f"{path}: checkpoint={saved!r}, requested={requested!r}"]
    return []


def validate_resume_configuration(checkpoint: dict[str, Any], config: dict[str, Any]) -> bool:
    """Validate trajectory-defining fields and report legacy provenance migration."""

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint does not contain a resolved configuration; safe resume is impossible")
    saved_config = copy.deepcopy(saved_config)
    migrated_legacy_provenance = "data_provenance" not in saved_config
    if migrated_legacy_provenance:
        # Older checkpoints still bind the exact data paths. Validate their current
        # sidecars now, then persist that provenance in the next resolved config and
        # checkpoint so subsequent resumes are fully self-describing.
        saved_config["data_provenance"] = copy.deepcopy(config["data_provenance"])

    saved_immutable = _resume_immutable_config(saved_config)
    requested_immutable = _resume_immutable_config(config)
    differences = _config_differences(saved_immutable, requested_immutable)
    if differences:
        preview = "; ".join(differences[:8])
        if len(differences) > 8:
            preview += f"; ... and {len(differences) - 8} more"
        raise ValueError(f"resume would change immutable configuration: {preview}")
    return migrated_legacy_provenance


def validate_checkpoint_optimizer_hyperparameters(checkpoint: dict[str, Any]) -> None:
    saved_config = checkpoint["config"]
    optimizer_config = saved_config.get("optimizer")
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_config, dict) or not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint is missing optimizer configuration or state")
    parameter_groups = optimizer_state.get("param_groups")
    if not isinstance(parameter_groups, list) or not parameter_groups:
        raise ValueError("checkpoint optimizer has no parameter groups")

    expected_betas = tuple(float(value) for value in optimizer_config["betas"])
    expected_eps = float(optimizer_config["eps"])
    expected_weight_decay = float(optimizer_config["weight_decay"])
    for index, group in enumerate(parameter_groups):
        if not isinstance(group, dict):
            raise ValueError(f"checkpoint optimizer parameter group {index} is malformed")
        actual_betas = tuple(float(value) for value in group.get("betas", ()))
        if actual_betas != expected_betas:
            raise ValueError(f"checkpoint optimizer parameter group {index} betas contradict checkpoint config")
        if float(group.get("eps", math.nan)) != expected_eps:
            raise ValueError(f"checkpoint optimizer parameter group {index} eps contradicts checkpoint config")
        if float(group.get("weight_decay", math.nan)) != expected_weight_decay:
            raise ValueError(f"checkpoint optimizer parameter group {index} weight_decay contradicts checkpoint config")


def apply_optimizer_hyperparameters(optimizer: AdamW, config: dict[str, Any]) -> None:
    """Keep loaded param groups consistent with the validated resolved config."""

    optimizer_config = config["optimizer"]
    betas = tuple(float(value) for value in optimizer_config["betas"])
    epsilon = float(optimizer_config["eps"])
    weight_decay = float(optimizer_config["weight_decay"])
    for group in optimizer.param_groups:
        group["betas"] = betas
        group["eps"] = epsilon
        group["weight_decay"] = weight_decay


def _event_progress(event: dict[str, Any], *, path: Path, line_number: int) -> tuple[int, int]:
    step = event.get("step")
    processed_tokens = event.get("processed_tokens")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"metric event at {path}:{line_number} has an invalid step")
    if not isinstance(processed_tokens, int) or isinstance(processed_tokens, bool) or processed_tokens < 0:
        raise ValueError(f"metric event at {path}:{line_number} has invalid processed_tokens")
    return step, processed_tokens


def prepare_metrics_for_resume(
    path: Path,
    *,
    checkpoint_step: int,
    checkpoint_processed_tokens: int,
) -> tuple[int, int | None, int | None]:
    """Atomically discard an uncheckpointed JSONL tail and return its frontier."""

    if not path.exists():
        return 0, None, None

    lines = path.read_text(encoding="utf-8").splitlines()
    retained: list[dict[str, Any]] = []
    previous_step: int | None = None
    previous_tokens: int | None = None
    dropped = 0
    for index, line in enumerate(lines):
        line_number = index + 1
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            if not any(remaining.strip() for remaining in lines[index + 1 :]):
                dropped += 1
                break
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        step, processed_tokens = _event_progress(event, path=path, line_number=line_number)
        if step > checkpoint_step or processed_tokens > checkpoint_processed_tokens:
            dropped += sum(1 for remaining in lines[index:] if remaining.strip())
            break
        if previous_step is not None and (step < previous_step or processed_tokens < previous_tokens):
            raise ValueError(
                f"metrics are already non-monotonic at {path}:{line_number}; "
                "start a new output directory or repair the log explicitly"
            )
        retained.append(event)
        previous_step = step
        previous_tokens = processed_tokens

    temporary = path.with_name(f".{path.name}.{os.getpid()}.resume.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for event in retained:
                output_file.write(strict_json_dumps(event) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dropped, previous_step, previous_tokens


class JSONLLogger:
    def __init__(
        self,
        path: Path,
        *,
        previous_step: int | None = None,
        previous_processed_tokens: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._previous_step = previous_step
        self._previous_processed_tokens = previous_processed_tokens
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def log(self, payload: dict[str, Any]) -> None:
        step, processed_tokens = _event_progress(payload, path=self._path, line_number=0)
        if self._previous_step is not None and (
            step < self._previous_step or processed_tokens < self._previous_processed_tokens
        ):
            raise ValueError(
                f"refusing to append non-monotonic metric progress ({step}, {processed_tokens}) "
                f"after ({self._previous_step}, {self._previous_processed_tokens})"
            )
        self._file.write(strict_json_dumps(payload) + "\n")
        self._previous_step = step
        self._previous_processed_tokens = processed_tokens

    def close(self) -> None:
        self._file.close()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def managed_run_setup(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    managed = ("metrics.jsonl", "config.resolved.json", "latest.pt", "diverged.pt")
    existing = [output_dir / name for name in managed if (output_dir / name).exists()]
    if existing and not resume and not overwrite:
        raise FileExistsError(f"run output already exists in {output_dir}; pass --resume or --overwrite")
    if overwrite and not resume:
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(args.config), DEFAULTS)
    apply_cli_overrides(config, args)
    validate_config(config)

    output_dir = Path(config["output_dir"])
    managed_run_setup(output_dir, resume=args.resume is not None, overwrite=args.overwrite)

    device = resolve_device(str(config["device"]))
    precision = str(config["precision"])
    seed = int(config["seed"])
    seed_everything(seed, bool(config["deterministic"]))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(config["allow_tf32"])

    context_length = int(config["model"]["context_length"])
    train_data = load_token_array(config["data"]["train"], context_length)
    validation_path = config["data"].get("validation")
    validation_data = load_token_array(validation_path, context_length) if validation_path else None
    data_provenance = attach_data_provenance(config, train_data, validation_data)

    model = TransformerLM(**config["model"], device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer_config = config["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )

    start_step = 0
    processed_tokens = 0
    previous_wall_time = 0.0
    migrated_legacy_provenance = False
    micro_batch_size = int(config["training"]["micro_batch_size"])
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    tokens_per_step = micro_batch_size * accumulation_steps * context_length
    metrics_path = output_dir / "metrics.jsonl"
    dropped_metric_events = 0
    previous_metric_step: int | None = None
    previous_metric_tokens: int | None = None
    if args.resume is not None:
        checkpoint_path = output_dir / "latest.pt" if args.resume == "auto" else Path(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("training resume requires a dictionary checkpoint")
        migrated_legacy_provenance = validate_resume_configuration(checkpoint, config)
        validate_checkpoint_optimizer_hyperparameters(checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        apply_optimizer_hyperparameters(optimizer, config)
        start_step = int(checkpoint["iteration"])
        processed_tokens = int(checkpoint["processed_tokens"])
        previous_wall_time = float(checkpoint.get("wall_time_seconds", 0.0))
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])

    max_steps = int(config["training"]["max_steps"])
    if start_step > max_steps:
        raise ValueError(f"checkpoint step {start_step} is beyond configured max_steps {max_steps}")
    if processed_tokens != start_step * tokens_per_step:
        raise ValueError(
            f"checkpoint processed_tokens {processed_tokens} does not equal "
            f"iteration * tokens_per_step ({start_step * tokens_per_step})"
        )
    if not math.isfinite(previous_wall_time) or previous_wall_time < 0:
        raise ValueError("checkpoint wall_time_seconds must be finite and non-negative")

    if args.resume is not None:
        dropped_metric_events, previous_metric_step, previous_metric_tokens = prepare_metrics_for_resume(
            metrics_path,
            checkpoint_step=start_step,
            checkpoint_processed_tokens=processed_tokens,
        )

    atomic_write_json(output_dir / "config.resolved.json", config)

    logger = JSONLLogger(
        metrics_path,
        previous_step=previous_metric_step,
        previous_processed_tokens=previous_metric_tokens,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_start = time.perf_counter()
    last_log_time = run_start
    last_log_tokens = processed_tokens
    latest_checkpoint = output_dir / "latest.pt"

    logger.log(
        {
            "event": "run_start",
            "timestamp_utc": utc_timestamp(),
            "run_name": str(config.get("run_name", output_dir.name)),
            "step": start_step,
            "processed_tokens": processed_tokens,
            "device_type": device.type,
            "precision": precision,
            "parameter_count": parameter_count,
            "effective_batch_size": micro_batch_size * accumulation_steps,
            "resumed": args.resume is not None,
            "dropped_metric_events": dropped_metric_events,
            "migrated_legacy_data_provenance": migrated_legacy_provenance,
            "tokenizer_sha256": data_provenance["train"]["tokenizer_sha256"],
            **device_memory_metrics(device),
        }
    )

    def abort_before_optimizer_step(
        *,
        reason: str,
        failed_step: int,
        loss_value: float | None,
        gradient_norm_value: float | None,
        micro_batch_index: int | None,
    ) -> None:
        wall_time = previous_wall_time + time.perf_counter() - run_start
        diagnostics = {
            "reason": reason,
            "failed_step": failed_step,
            "micro_batch_index": micro_batch_index,
            "loss": loss_value,
            "gradient_norm": gradient_norm_value,
            "optimizer_step_applied": False,
        }
        save_training_checkpoint(
            output_dir / "diverged.pt",
            model=model,
            optimizer=optimizer,
            step=failed_step,
            processed_tokens=processed_tokens,
            wall_time_seconds=wall_time,
            config=config,
            status="diverged",
            diagnostics=diagnostics,
        )
        logger.log(
            {
                "event": "diverged",
                "step": failed_step,
                "processed_tokens": processed_tokens,
                "reason": reason,
                "loss": loss_value,
                "gradient_norm": gradient_norm_value,
                "micro_batch_index": micro_batch_index,
                "optimizer_step_applied": False,
                "wall_time_seconds": wall_time,
            }
        )
        raise FloatingPointError(
            f"training diverged before optimizer step {failed_step}: "
            f"{reason} (loss={loss_value}, gradient_norm={gradient_norm_value})"
        )

    try:
        if validation_data is not None and bool(config["evaluation"]["at_start"]) and start_step == 0:
            synchronize_device(device)
            validation_start = time.perf_counter()
            validation_loss = evaluate(
                model,
                validation_data,
                batches=int(config["evaluation"]["batches"]),
                batch_size=int(config["evaluation"]["batch_size"]),
                context_length=context_length,
                device=device,
                precision=precision,
                evaluation_seed=seed + 1_000_003,
            )
            synchronize_device(device)
            logger.log(
                {
                    "event": "validation",
                    "step": 0,
                    "processed_tokens": 0,
                    "loss": validation_loss,
                    "wall_time_seconds": previous_wall_time + time.perf_counter() - run_start,
                    "evaluation_seconds": time.perf_counter() - validation_start,
                }
            )

        model.train()
        for step_index in range(start_step, max_steps):
            learning_rate = get_lr_cosine_schedule(
                step_index,
                max_learning_rate=float(optimizer_config["learning_rate"]),
                min_learning_rate=float(config["schedule"]["min_learning_rate"]),
                warmup_iters=int(config["schedule"]["warmup_iters"]),
                cosine_cycle_iters=int(config["schedule"]["cosine_cycle_iters"] or max_steps),
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for micro_batch_index in range(accumulation_steps):
                inputs, targets = get_batch(train_data, micro_batch_size, context_length, device)
                with precision_context(device, precision):
                    unscaled_loss = cross_entropy(model(inputs), targets)
                    loss = unscaled_loss / accumulation_steps
                detached_loss = float(unscaled_loss.detach().float().item())
                if not math.isfinite(detached_loss):
                    abort_before_optimizer_step(
                        reason="nonfinite_loss",
                        failed_step=step_index,
                        loss_value=detached_loss,
                        gradient_norm_value=None,
                        micro_batch_index=micro_batch_index,
                    )
                loss.backward()
                accumulated_loss += detached_loss

            completed_step = step_index + 1
            should_log = completed_step % int(config["logging"]["interval"]) == 0 or completed_step == max_steps
            global_gradient_norm = gradient_norm(model)
            if not math.isfinite(global_gradient_norm):
                abort_before_optimizer_step(
                    reason="nonfinite_gradient_norm",
                    failed_step=step_index,
                    loss_value=accumulated_loss / accumulation_steps,
                    gradient_norm_value=global_gradient_norm,
                    micro_batch_index=None,
                )
            gradient_clipping(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            processed_tokens += tokens_per_step

            if should_log:
                synchronize_device(device)
                now = time.perf_counter()
                elapsed_since_log = now - last_log_time
                logger.log(
                    {
                        "event": "train",
                        "step": completed_step,
                        "processed_tokens": processed_tokens,
                        "loss": accumulated_loss / accumulation_steps,
                        "learning_rate": learning_rate,
                        "gradient_norm": global_gradient_norm,
                        "wall_time_seconds": previous_wall_time + now - run_start,
                        "tokens_per_second": (processed_tokens - last_log_tokens) / elapsed_since_log,
                        **device_memory_metrics(device),
                    }
                )
                last_log_time = now
                last_log_tokens = processed_tokens

            if validation_data is not None and completed_step % int(config["evaluation"]["interval"]) == 0:
                synchronize_device(device)
                validation_start = time.perf_counter()
                validation_loss = evaluate(
                    model,
                    validation_data,
                    batches=int(config["evaluation"]["batches"]),
                    batch_size=int(config["evaluation"]["batch_size"]),
                    context_length=context_length,
                    device=device,
                    precision=precision,
                    evaluation_seed=seed + 1_000_003 + completed_step,
                )
                synchronize_device(device)
                logger.log(
                    {
                        "event": "validation",
                        "step": completed_step,
                        "processed_tokens": processed_tokens,
                        "loss": validation_loss,
                        "wall_time_seconds": previous_wall_time + time.perf_counter() - run_start,
                        "evaluation_seconds": time.perf_counter() - validation_start,
                    }
                )

            if completed_step % int(config["checkpoint"]["interval"]) == 0:
                wall_time = previous_wall_time + time.perf_counter() - run_start
                save_training_checkpoint(
                    latest_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=completed_step,
                    processed_tokens=processed_tokens,
                    wall_time_seconds=wall_time,
                    config=config,
                    status="running",
                )
                logger.log(
                    {
                        "event": "checkpoint",
                        "step": completed_step,
                        "processed_tokens": processed_tokens,
                        "wall_time_seconds": wall_time,
                        "filename": latest_checkpoint.name,
                    }
                )

        final_wall_time = previous_wall_time + time.perf_counter() - run_start
        save_training_checkpoint(
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            step=max_steps,
            processed_tokens=processed_tokens,
            wall_time_seconds=final_wall_time,
            config=config,
            status="complete",
        )
        logger.log(
            {
                "event": "run_end",
                "timestamp_utc": utc_timestamp(),
                "step": max_steps,
                "processed_tokens": processed_tokens,
                "wall_time_seconds": final_wall_time,
                "status": "complete",
                **device_memory_metrics(device),
            }
        )
    finally:
        logger.close()

    print(
        json.dumps(
            {
                "status": "complete",
                "step": max_steps,
                "processed_tokens": processed_tokens,
                "wall_time_seconds": final_wall_time,
                "checkpoint": str(latest_checkpoint),
                "metrics": str(output_dir / "metrics.jsonl"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
