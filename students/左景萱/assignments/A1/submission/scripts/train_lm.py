#!/usr/bin/env python3
"""Train the assignment Transformer LM from a JSON configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.model import TransformerLM  # noqa: E402
from cs336_basics.training import (  # noqa: E402
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key; VALUE is parsed as JSON when possible.",
    )
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_runtime_environment() -> None:
    """Keep compilation caches and temporary files next to the project."""

    runtime_root = ROOT / ".runtime"
    locations = {
        "HOME": runtime_root / "home",
        "XDG_CACHE_HOME": runtime_root / "cache",
        "TMPDIR": runtime_root / "tmp",
        "TMP": runtime_root / "tmp",
        "TEMP": runtime_root / "tmp",
        "PYTHONPYCACHEPREFIX": runtime_root / "pycache",
        "TORCHINDUCTOR_CACHE_DIR": runtime_root / "torchinductor",
        "TRITON_CACHE_DIR": runtime_root / "triton",
        "UV_CACHE_DIR": runtime_root / "uv-cache",
        "TORCH_HOME": runtime_root / "torch",
        "CUDA_CACHE_PATH": runtime_root / "cuda",
        "TORCH_EXTENSIONS_DIR": runtime_root / "torch-extensions",
    }
    project_root = ROOT.resolve()
    for variable, default_directory in locations.items():
        configured = os.environ.get(variable)
        directory = Path(configured) if configured else default_directory
        if not directory.is_absolute():
            directory = ROOT / directory
        directory = directory.resolve(strict=False)
        try:
            directory.relative_to(project_root)
        except ValueError:
            directory = default_directory.resolve(strict=False)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must have KEY=VALUE form: {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dotted(config, key, _parse_value(raw_value))
    return config


def resolve_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def seed_everything(seed: int, *, seed_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def load_token_array(path: str, dtype: str) -> np.memmap:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"token data not found: {resolved}")
    array = np.memmap(resolved, dtype=np.dtype(dtype), mode="r")
    if array.ndim != 1:
        raise ValueError(f"expected a flat token array at {resolved}")
    return array


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unsupported precision: {precision}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.memmap,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
    precision: str,
    num_batches: int,
    validation_seed: int,
) -> float:
    was_training = model.training
    model.eval()
    numpy_state = np.random.get_state()
    np.random.seed(validation_seed)
    losses: list[float] = []
    try:
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            with autocast_context(device, precision):
                logits = model(inputs)
                loss = cross_entropy(logits.float(), targets)
            losses.append(float(loss))
    finally:
        np.random.set_state(numpy_state)
        model.train(was_training)
    return sum(losses) / len(losses)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_metric(file, metric: dict[str, Any]) -> None:
    file.write(json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n")
    file.flush()


def read_existing_run_state(output_dir: Path) -> tuple[float, float, float]:
    """Recover cumulative log state when appending to a resumed run."""

    elapsed_seconds = 0.0
    best_validation_loss = math.inf
    final_training_loss = math.nan

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            elapsed_seconds = max(elapsed_seconds, float(summary.get("total_training_time_sec", 0.0)))
            best_validation_loss = min(best_validation_loss, float(summary.get("best_val_loss", math.inf)))
            final_training_loss = float(summary.get("final_train_loss", math.nan))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                metric = json.loads(line)
                elapsed_seconds = max(elapsed_seconds, float(metric.get("wall_clock_sec", 0.0)))
                if "val_loss" in metric:
                    best_validation_loss = min(best_validation_loss, float(metric["val_loss"]))
                if "train_loss" in metric:
                    final_training_loss = float(metric["train_loss"])
            except (json.JSONDecodeError, TypeError, ValueError):
                # A process interrupted during a write can leave one partial line.
                continue

    return elapsed_seconds, best_validation_loss, final_training_loss


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_everything(seed, seed_cuda=device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data_config = config["data"]
    model_config = deepcopy(config["model"])
    training_config = config["training"]
    train_data = load_token_array(data_config["train_path"], data_config.get("dtype", "uint16"))
    validation_data = load_token_array(data_config["val_path"], data_config.get("dtype", "uint16"))

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and args.resume is None and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; use --overwrite or --resume")
    if args.overwrite and args.resume is None:
        for stale_name in ("metrics.jsonl", "summary.json", "best.pt", "latest.pt", "final.pt"):
            (output_dir / stale_name).unlink(missing_ok=True)

    write_json(output_dir / "resolved_config.json", config)
    model = TransformerLM(**model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["max_lr"]),
        betas=tuple(training_config.get("betas", (0.9, 0.95))),
        eps=float(training_config.get("eps", 1e-8)),
        weight_decay=float(training_config.get("weight_decay", 0.1)),
    )

    start_iteration = 0
    if args.resume is not None:
        start_iteration = load_checkpoint(resolve_path(args.resume), model, optimizer)

    precision = training_config.get("precision", "bf16")
    train_model: torch.nn.Module = model
    if bool(training_config.get("compile", False)) and hasattr(torch, "compile"):
        train_model = torch.compile(model)

    batch_size = int(training_config["batch_size"])
    context_length = int(model_config["context_length"])
    max_steps = int(training_config["max_steps"])
    warmup_steps = int(training_config.get("warmup_steps", 0))
    max_lr = float(training_config["max_lr"])
    min_lr = float(training_config.get("min_lr", 0.0))
    log_every = int(training_config.get("log_every", 10))
    val_every = int(training_config.get("val_every", 250))
    val_batches = int(training_config.get("val_batches", 20))
    checkpoint_every = int(training_config.get("checkpoint_every", 1000))
    max_grad_norm = float(training_config.get("grad_clip", 1.0))

    run_name = str(config.get("run_name", output_dir.name))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    training_start = time.perf_counter()
    elapsed_before_resume = 0.0
    best_validation_loss = math.inf
    final_training_loss = math.nan
    if args.resume is not None:
        elapsed_before_resume, best_validation_loss, final_training_loss = read_existing_run_state(output_dir)
    last_validation_loss = math.nan
    completed_steps = start_iteration
    status = "completed"
    divergence_reason: str | None = None

    mode = "a" if args.resume is not None else "w"
    with metrics_path.open(mode, encoding="utf-8") as metrics_file:
        initial_validation_loss = evaluate(
            train_model,
            validation_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            precision=precision,
            num_batches=val_batches,
            validation_seed=seed + 1,
        )
        last_validation_loss = initial_validation_loss
        if initial_validation_loss < best_validation_loss or not (output_dir / "best.pt").is_file():
            save_checkpoint(model, optimizer, start_iteration, output_dir / "best.pt")
        best_validation_loss = min(best_validation_loss, initial_validation_loss)
        append_metric(
            metrics_file,
            {
                "event": "validation",
                "step": start_iteration,
                "wall_clock_sec": elapsed_before_resume,
                "val_loss": initial_validation_loss,
                "lr": 0.0,
                "processed_tokens": start_iteration * batch_size * context_length,
            },
        )

        model.train()
        for iteration in range(start_iteration, max_steps):
            learning_rate = get_lr_cosine_schedule(
                iteration,
                max_lr,
                min_lr,
                warmup_steps,
                max_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            inputs, targets = get_batch(train_data, batch_size, context_length, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = train_model(inputs)
                loss = cross_entropy(logits.float(), targets)

            if not bool(torch.isfinite(loss)):
                status = "diverged"
                divergence_reason = "non_finite_loss"
                final_training_loss = float(loss)
                completed_steps = iteration
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                append_metric(
                    metrics_file,
                    {
                        "event": "diverged",
                        "reason": divergence_reason,
                        "step": iteration,
                        "wall_clock_sec": elapsed_before_resume + time.perf_counter() - training_start,
                        "train_loss": final_training_loss,
                        "lr": learning_rate,
                        "processed_tokens": iteration * batch_size * context_length,
                    },
                )
                break

            loss.backward()
            try:
                gradient_clipping(model.parameters(), max_grad_norm)
            except FloatingPointError:
                status = "diverged"
                divergence_reason = "non_finite_gradient"
                final_training_loss = float(loss.detach())
                completed_steps = iteration
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                append_metric(
                    metrics_file,
                    {
                        "event": "diverged",
                        "reason": divergence_reason,
                        "step": iteration,
                        "wall_clock_sec": elapsed_before_resume + time.perf_counter() - training_start,
                        "train_loss": final_training_loss,
                        "lr": learning_rate,
                        "processed_tokens": iteration * batch_size * context_length,
                    },
                )
                break
            optimizer.step()
            completed_steps = iteration + 1
            final_training_loss = float(loss.detach())

            should_log = completed_steps % log_every == 0 or completed_steps == 1
            should_validate = completed_steps % val_every == 0 or completed_steps == max_steps
            if should_validate:
                last_validation_loss = evaluate(
                    train_model,
                    validation_data,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                    precision=precision,
                    num_batches=val_batches,
                    validation_seed=seed + 1,
                )
                if last_validation_loss < best_validation_loss:
                    best_validation_loss = last_validation_loss
                    save_checkpoint(model, optimizer, completed_steps, output_dir / "best.pt")

            if should_log or should_validate:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                metric = {
                    "event": "train",
                    "step": completed_steps,
                    "wall_clock_sec": elapsed_before_resume + time.perf_counter() - training_start,
                    "train_loss": final_training_loss,
                    "lr": learning_rate,
                    "processed_tokens": completed_steps * batch_size * context_length,
                }
                if should_validate:
                    metric["val_loss"] = last_validation_loss
                append_metric(metrics_file, metric)

            if checkpoint_every > 0 and completed_steps % checkpoint_every == 0:
                save_checkpoint(model, optimizer, completed_steps, output_dir / "latest.pt")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = elapsed_before_resume + time.perf_counter() - training_start
    save_checkpoint(model, optimizer, completed_steps, output_dir / "final.pt")

    summary = {
        "run_name": run_name,
        "status": status,
        "divergence_reason": divergence_reason,
        "completed_steps": completed_steps,
        "processed_tokens": completed_steps * batch_size * context_length,
        "total_training_time_sec": total_seconds,
        "final_train_loss": final_training_loss,
        "final_val_loss": last_validation_loss,
        "best_val_loss": best_validation_loss if math.isfinite(best_validation_loss) else last_validation_loss,
        "parameter_count": parameter_count,
        "device_type": device.type,
        "model": model_config,
        "training": training_config,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> int:
    args = parse_args()
    configure_runtime_environment()
    config = load_config(args.config, args.set)
    train(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
