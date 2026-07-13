"""Data loading, checkpointing, and a reproducible language-model trainer."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, IO, cast

import numpy as np
import torch
from torch import Tensor

from cs336_basics.optim import AdamW, clip_gradients, cross_entropy, get_lr_cosine_schedule, global_gradient_norm


PathOrFile = str | os.PathLike[str] | BinaryIO | IO[bytes]


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample next-token prediction examples from a 1D token array.

    Sampling indices are created on CPU even when the returned tensors live on
    an accelerator.  This keeps the function compatible with memory-mapped
    NumPy arrays and makes its RNG state easy to checkpoint.
    """

    if getattr(dataset, "ndim", None) != 1:
        raise ValueError("dataset must be a one-dimensional token array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError(
            f"dataset must contain at least context_length + 1 tokens; got {len(dataset)} and {context_length}"
        )

    starts = torch.randint(0, num_starts, (batch_size,), generator=generator, device="cpu").numpy()
    offsets = np.arange(context_length + 1, dtype=np.int64)
    token_indices = starts[:, None] + offsets[None, :]
    # astype(copy=True) avoids retaining a view into a read-only mmap and
    # converts compact uint16 corpora to the dtype required by embeddings.
    batch = np.asarray(dataset[token_indices]).astype(np.int64, copy=True)
    batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.long)
    return batch_tensor[:, :-1], batch_tensor[:, 1:]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: PathOrFile,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Serialize model, optimizer, iteration, and optional resumability data."""

    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    if extra is not None:
        payload["extra"] = dict(extra)
    torch.save(payload, out)


def _read_checkpoint(src: PathOrFile, map_location: str | torch.device | None = None) -> dict[str, Any]:
    # Checkpoints are local, trusted training artifacts and can contain Python
    # RNG tuples in addition to tensors, hence weights_only=False is explicit.
    payload = torch.load(src, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload


def _checkpoint_value(payload: Mapping[str, Any], primary: str, fallback: str) -> Any:
    if primary in payload:
        return payload[primary]
    if fallback in payload:
        return payload[fallback]
    raise KeyError(f"checkpoint is missing {primary!r}")


def load_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state and return the saved iteration."""

    payload = _read_checkpoint(src)
    model.load_state_dict(_checkpoint_value(payload, "model_state_dict", "model"))
    optimizer.load_state_dict(_checkpoint_value(payload, "optimizer_state_dict", "optimizer"))
    return int(payload["iteration"])


def load_training_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[int, dict[str, Any]]:
    """Restore a checkpoint and also return optional training-loop metadata."""

    payload = _read_checkpoint(src, map_location=map_location)
    model.load_state_dict(_checkpoint_value(payload, "model_state_dict", "model"))
    optimizer.load_state_dict(_checkpoint_value(payload, "optimizer_state_dict", "optimizer"))
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("checkpoint 'extra' field must be a dictionary")
    return int(payload["iteration"]), extra


def load_model_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[int, dict[str, Any]]:
    """Load only model state, as needed for inference."""

    payload = _read_checkpoint(src, map_location=map_location)
    model.load_state_dict(_checkpoint_value(payload, "model_state_dict", "model"))
    extra = payload.get("extra", {})
    return int(payload.get("iteration", 0)), extra if isinstance(extra, dict) else {}


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a TOML experiment configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(config_path)
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable hash while excluding machine-specific config metadata."""

    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    serialized = json.dumps(public_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"[{name}] must be a TOML table")
    return dict(value)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _resolve_input_path(value: str | os.PathLike[str], config: Mapping[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    if config_path := config.get("_config_path"):
        config_dir = Path(str(config_path)).parent
        candidates.extend((config_dir / path, config_dir.parent / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_output_path(value: str | os.PathLike[str], config: Mapping[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    run_cfg = _section(config, "run")
    if root := run_cfg.get("root_dir"):
        return (Path(str(root)).expanduser() / path).resolve()
    return (Path.cwd() / path).resolve()


def load_token_array(
    path: str | os.PathLike[str],
    *,
    dtype: str | np.dtype[Any] = "uint16",
) -> np.ndarray:
    """Memory-map either a ``.npy`` array or a raw compact token file."""

    data_path = Path(path)
    if data_path.suffix == ".npy":
        array = np.load(data_path, mmap_mode="r", allow_pickle=False)
    else:
        array = np.memmap(data_path, mode="r", dtype=np.dtype(dtype))
    if array.ndim != 1:
        raise ValueError(f"token array must be 1D, got shape {array.shape}")
    if len(array) < 2:
        raise ValueError("token array must contain at least two tokens")
    return array


def resolve_device(requested: str | None) -> torch.device:
    """Resolve ``auto`` while rejecting unavailable accelerator requests."""

    requested = requested or "auto"
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return device


def _torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None or name in {"default", "auto"}:
        return None
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype: {name}") from error


def build_model(config: Mapping[str, Any], device: str | torch.device | None = None) -> torch.nn.Module:
    """Construct ``TransformerLM`` from the ``[model]`` TOML table."""

    from cs336_basics.transformer import TransformerLM

    model_cfg = _section(config, "model")
    required = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff")
    missing = [name for name in required if name not in model_cfg]
    if missing:
        raise KeyError(f"[model] is missing required keys: {', '.join(missing)}")

    target_device = torch.device(device) if device is not None else resolve_device(_section(config, "training").get("device"))
    dtype = _torch_dtype(model_cfg.pop("dtype", None))
    allowed = {
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "rope_theta",
        "remove_rmsnorm",
        "use_post_norm",
        "remove_rope",
        "ffn_type",
        "silu_d_ff",
    }
    unexpected = sorted(set(model_cfg) - allowed)
    if unexpected:
        raise KeyError(f"unsupported [model] keys: {', '.join(unexpected)}")
    return TransformerLM(**model_cfg, device=target_device, dtype=dtype)


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> AdamW:
    """Build AdamW, separating matrix weights from norm/bias parameters."""

    optimizer_cfg = _section(config, "optimizer")
    schedule_cfg = _section(config, "schedule")
    lr = float(_first(optimizer_cfg, "lr", default=_first(schedule_cfg, "max_learning_rate", "max_lr", default=1e-3)))
    betas_value = optimizer_cfg.get("betas", (optimizer_cfg.get("beta1", 0.9), optimizer_cfg.get("beta2", 0.999)))
    betas = (float(betas_value[0]), float(betas_value[1]))
    eps = float(optimizer_cfg.get("eps", 1e-8))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    separate = bool(optimizer_cfg.get("separate_weight_decay", True))

    if not separate or weight_decay == 0:
        return AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay_params if parameter.ndim >= 2 else no_decay_params).append(parameter)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state(train_generator: torch.Generator, validation_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
        "validation_generator": validation_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: Mapping[str, Any], train_generator: torch.Generator, validation_generator: torch.Generator
) -> None:
    def _as_cpu_byte(value: Any) -> torch.Tensor:
        # ``torch.load`` with a CUDA ``map_location`` moves the saved RNG state
        # tensors onto the GPU, but ``set_rng_state`` / ``Generator.set_state``
        # require a CPU ByteTensor.  Coerce back so resume works across devices.
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.detach().to(device="cpu", dtype=torch.uint8)

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_byte(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_cpu_byte(s) for s in state["cuda"]])
    if "train_generator" in state:
        train_generator.set_state(_as_cpu_byte(state["train_generator"]))
    if "validation_generator" in state:
        validation_generator.set_state(_as_cpu_byte(state["validation_generator"]))


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    num_batches: int,
    generator: torch.Generator | None = None,
) -> float:
    """Estimate mean per-token validation loss over random batches."""

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        inputs, targets = get_batch(
            dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            generator=generator,
        )
        total += float(cross_entropy(model(inputs), targets).detach().cpu())
    model.train(was_training)
    return total / num_batches


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _memory_metrics(device: torch.device) -> dict[str, int]:
    metrics: dict[str, int] = {}
    try:
        import psutil

        metrics["rss_bytes"] = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass
    if device.type == "cuda":
        metrics["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return metrics


def _atomic_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    extra: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        save_checkpoint(model, optimizer, iteration, temporary, extra=extra)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_from_config(
    config_or_path: Mapping[str, Any] | str | os.PathLike[str],
    *,
    resume: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run a configurable LM training job and return its final summary.

    Expected TOML tables are ``[model]``, ``[data]``, ``[optimizer]``,
    ``[schedule]``, ``[training]``, and optionally ``[run]``.  Token arrays are
    opened in memory-mapped mode and metrics are written as JSONL.
    """

    config = dict(config_or_path) if isinstance(config_or_path, Mapping) else load_config(config_or_path)
    training_cfg = _section(config, "training")
    data_cfg = _section(config, "data")
    schedule_cfg = _section(config, "schedule")
    run_cfg = _section(config, "run")

    seed = int(training_cfg.get("seed", 1337))
    _set_seed(seed)
    device = resolve_device(training_cfg.get("device"))
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)

    context_length = int(_section(config, "model")["context_length"])
    batch_size = int(training_cfg.get("batch_size", 1))
    max_iters = int(_first(training_cfg, "max_iters", "max_steps", default=1))
    if max_iters <= 0:
        raise ValueError("training.max_iters must be positive")

    train_path_value = _first(data_cfg, "train_path", "train")
    validation_path_value = _first(data_cfg, "validation_path", "val_path", "validation", "val")
    if train_path_value is None or validation_path_value is None:
        raise KeyError("[data] must define train_path and validation_path")
    token_dtype = data_cfg.get("dtype", "uint16")
    train_data = load_token_array(_resolve_input_path(train_path_value, config), dtype=token_dtype)
    validation_data = load_token_array(_resolve_input_path(validation_path_value, config), dtype=token_dtype)

    vocab_size = int(_section(config, "model")["vocab_size"])
    if bool(data_cfg.get("validate_token_range", True)):
        # Full scans defeat mmap for very large corpora, so inspect evenly
        # spaced samples unless strict validation is explicitly requested.
        for name, array in (("train", train_data), ("validation", validation_data)):
            if bool(data_cfg.get("strict_token_range", False)):
                observed_max = int(np.max(array))
                observed_min = int(np.min(array))
            else:
                sample_indices = np.linspace(0, len(array) - 1, min(len(array), 4096), dtype=np.int64)
                sample = array[sample_indices]
                observed_max, observed_min = int(np.max(sample)), int(np.min(sample))
            if observed_min < 0 or observed_max >= vocab_size:
                raise ValueError(
                    f"{name} token sample is outside [0, {vocab_size}): min={observed_min}, max={observed_max}"
                )

    output_dir = _resolve_output_path(run_cfg.get("output_dir", training_cfg.get("output_dir", "runs/default")), config)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / str(run_cfg.get("log_file", "metrics.jsonl"))
    checkpoint_path = output_dir / str(run_cfg.get("checkpoint_file", "latest.pt"))
    experiment_hash = config_hash(config)

    train_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    validation_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    start_iteration = 0
    resume_value = resume or training_cfg.get("resume")
    if resume_value:
        resolved_resume = checkpoint_path if str(resume_value) == "latest" else _resolve_input_path(resume_value, config)
        start_iteration, extra = load_training_checkpoint(resolved_resume, model, optimizer, map_location=device)
        if rng := extra.get("rng_state"):
            _restore_rng_state(rng, train_generator, validation_generator)
    if start_iteration > max_iters:
        raise ValueError(
            f"checkpoint iteration {start_iteration} exceeds configured max_iters {max_iters}"
        )

    forward_model: torch.nn.Module = model
    if bool(training_cfg.get("compile", False)):
        compile_kwargs: dict[str, Any] = {}
        if backend := training_cfg.get("compile_backend"):
            compile_kwargs["backend"] = backend
        forward_model = cast(torch.nn.Module, torch.compile(model, **compile_kwargs))

    max_lr = float(
        _first(
            schedule_cfg,
            "max_learning_rate",
            "max_lr",
            default=_section(config, "optimizer").get("lr", 1e-3),
        )
    )
    min_lr = float(_first(schedule_cfg, "min_learning_rate", "min_lr", default=max_lr))
    warmup_iters = int(schedule_cfg.get("warmup_iters", 0))
    cosine_cycle_iters = int(schedule_cfg.get("cosine_cycle_iters", max_iters))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    log_interval = int(training_cfg.get("log_interval", 10))
    eval_interval = int(training_cfg.get("eval_interval", 100))
    eval_iters = int(training_cfg.get("eval_iters", 10))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 500))
    if log_interval < 0 or eval_interval < 0 or checkpoint_interval < 0:
        raise ValueError("log/eval/checkpoint intervals must be non-negative")
    keep_step_checkpoints = bool(run_cfg.get("keep_step_checkpoints", False))
    started_at = time.monotonic()

    _append_jsonl(
        log_path,
        {
            "event": "run_start",
            "config_hash": experiment_hash,
            "device": str(device),
            "iteration": start_iteration,
            "max_iters": max_iters,
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "seed": seed,
            "time": time.time(),
        },
    )

    last_loss: float | None = None
    last_validation_loss: float | None = None
    if bool(training_cfg.get("eval_at_start", False)):
        initial_validation = estimate_loss(
            forward_model,
            validation_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            num_batches=eval_iters,
            generator=validation_generator,
        )
        last_validation_loss = initial_validation
        _append_jsonl(log_path, {"event": "validation", "iteration": start_iteration, "loss": initial_validation})

    for iteration in range(start_iteration, max_iters):
        learning_rate = get_lr_cosine_schedule(
            iteration,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(
            train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(inputs)
        loss = cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at iteration {iteration}: {float(loss.detach().cpu())}")
        loss.backward()
        grad_norm = float(global_gradient_norm(model.parameters()).detach().cpu())
        clip_gradients(model.parameters(), max_grad_norm)
        optimizer.step()

        completed = iteration + 1
        last_loss = float(loss.detach().cpu())
        elapsed = time.monotonic() - started_at
        should_log = (log_interval > 0 and completed % log_interval == 0) or completed == 1 or completed == max_iters
        if should_log:
            session_tokens = (completed - start_iteration) * batch_size * context_length
            record: dict[str, Any] = {
                "event": "train",
                "iteration": completed,
                "loss": last_loss,
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "processed_tokens": completed * batch_size * context_length,
                "tokens_per_second": session_tokens / max(elapsed, 1e-12),
                "wall_time_seconds": elapsed,
            }
            record.update(_memory_metrics(device))
            _append_jsonl(log_path, record)

        should_evaluate = (eval_interval > 0 and completed % eval_interval == 0) or completed == max_iters
        if should_evaluate:
            last_validation_loss = estimate_loss(
                forward_model,
                validation_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                num_batches=eval_iters,
                generator=validation_generator,
            )
            _append_jsonl(
                log_path,
                {
                    "event": "validation",
                    "iteration": completed,
                    "loss": last_validation_loss,
                    "wall_time_seconds": time.monotonic() - started_at,
                },
            )

        should_checkpoint = (checkpoint_interval > 0 and completed % checkpoint_interval == 0) or completed == max_iters
        if should_checkpoint:
            extra = {
                "config_hash": experiment_hash,
                "rng_state": _rng_state(train_generator, validation_generator),
            }
            _atomic_training_checkpoint(checkpoint_path, model, optimizer, completed, extra)
            if keep_step_checkpoints:
                step_path = output_dir / f"step_{completed:08d}.pt"
                _atomic_training_checkpoint(step_path, model, optimizer, completed, extra)
            _append_jsonl(log_path, {"event": "checkpoint", "iteration": completed, "file": checkpoint_path.name})

    summary = {
        "event": "run_end",
        "config_hash": experiment_hash,
        "iteration": max_iters,
        "loss": last_loss,
        "validation_loss": last_validation_loss,
        "processed_tokens": max_iters * batch_size * context_length,
        "wall_time_seconds": time.monotonic() - started_at,
        "checkpoint": checkpoint_path.name,
    }
    _append_jsonl(log_path, summary)
    return summary


# Conventional names used by some external A1 implementations.
get_lr = get_lr_cosine_schedule
train = train_from_config
