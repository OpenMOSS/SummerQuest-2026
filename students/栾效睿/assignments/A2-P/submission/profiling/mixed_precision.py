"""Task 3 accumulation and ToyModel dtype experiments.

Use the unified ``profiling/benchmark.py`` with ``--dtype bf16`` for language
model timing and memory measurements. This module records the two small,
diagnostic experiments required before that benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr
from profiling.benchmark import MODEL_SPECS, ModelConfig, build_model


DEFAULT_OUTPUT = Path("results/mixed_precision.json")


@dataclass(frozen=True)
class NumericTrendConfig:
    """Fixed, reproducible FP32-versus-BF16 diagnostic configuration."""

    model_size: str = "small"
    batch_size: int = 4
    context_length: int = 512
    vocab_size: int = 10_000
    seed: int = 0
    warmup_steps: int = 5
    measurement_steps: int = 10
    lr_max: float = 1e-3
    lr_min: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0


NUMERIC_METRIC_NAMES = (
    "fp32_loss",
    "bf16_loss",
    "loss_abs_diff",
    "loss_relative_diff",
    "logits_max_abs_diff",
    "logits_rmse",
    "logits_relative_l2_error",
    "top1_agreement",
)


def _json_number(value: torch.Tensor | float) -> float | None:
    """Convert a scalar to a JSON-safe number, preserving non-finite evidence as null."""

    scalar = float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)
    return scalar if math.isfinite(scalar) else None


def _all_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def numeric_error_metrics(
    *,
    fp32_logits: torch.Tensor,
    bf16_logits: torch.Tensor,
    fp32_loss: torch.Tensor,
    bf16_loss: torch.Tensor,
) -> dict[str, float | bool | None]:
    """Return JSON-safe FP32-versus-BF16 numerical-error metrics for one step.

    Inputs are deliberately tensors rather than model objects so the metric and
    summary code can be verified on CPU.  BF16 logits are promoted to FP32 for
    all error calculations; this avoids measuring the precision of the metric
    implementation itself.
    """

    if fp32_logits.shape != bf16_logits.shape:
        raise ValueError(f"Cannot compare logits with different shapes: {tuple(fp32_logits.shape)} and {tuple(bf16_logits.shape)}.")
    if fp32_loss.numel() != 1 or bf16_loss.numel() != 1:
        raise ValueError("FP32 and BF16 losses must each contain exactly one value.")

    fp32_logits_finite = _all_finite(fp32_logits)
    bf16_logits_finite = _all_finite(bf16_logits)
    fp32_loss_finite = _all_finite(fp32_loss)
    bf16_loss_finite = _all_finite(bf16_loss)
    all_finite = fp32_logits_finite and bf16_logits_finite and fp32_loss_finite and bf16_loss_finite

    fp32_logits_float = fp32_logits.detach().float()
    bf16_logits_float = bf16_logits.detach().float()
    logits_difference = bf16_logits_float - fp32_logits_float
    loss_difference = bf16_loss.detach().float() - fp32_loss.detach().float()
    epsilon = torch.finfo(torch.float32).eps

    # Top-1 agreement is only meaningful when both complete logits tensors are finite.
    top1_agreement: float | None = None
    if fp32_logits_finite and bf16_logits_finite:
        top1_agreement = _json_number((fp32_logits_float.argmax(dim=-1) == bf16_logits_float.argmax(dim=-1)).float().mean())

    return {
        "fp32_loss": _json_number(fp32_loss),
        "bf16_loss": _json_number(bf16_loss),
        "loss_abs_diff": _json_number(loss_difference.abs()),
        "loss_relative_diff": _json_number(loss_difference.abs() / fp32_loss.detach().float().abs().clamp_min(epsilon)),
        "logits_max_abs_diff": _json_number(logits_difference.abs().max()),
        "logits_rmse": _json_number(logits_difference.square().mean().sqrt()),
        "logits_relative_l2_error": _json_number(
            torch.linalg.vector_norm(logits_difference) / torch.linalg.vector_norm(fp32_logits_float).clamp_min(epsilon)
        ),
        "top1_agreement": top1_agreement,
        "fp32_loss_finite": fp32_loss_finite,
        "bf16_loss_finite": bf16_loss_finite,
        "fp32_logits_finite": fp32_logits_finite,
        "bf16_logits_finite": bf16_logits_finite,
        "all_finite": all_finite,
    }


def summarize_numeric_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a measurement sequence without emitting invalid JSON NaN values."""

    if not steps:
        raise ValueError("Numeric-trend summary requires at least one measurement step.")

    metrics: dict[str, dict[str, float | int | None]] = {}
    for name in NUMERIC_METRIC_NAMES:
        values = [float(value) for step in steps if isinstance((value := step.get(name)), int | float) and math.isfinite(float(value))]
        metrics[name] = {
            "min": min(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "max": max(values) if values else None,
            "available_steps": len(values),
        }

    finite_names = ("fp32_loss_finite", "bf16_loss_finite", "fp32_logits_finite", "bf16_logits_finite", "all_finite")
    return {
        "measurement_steps": len(steps),
        "all_steps_finite": all(bool(step.get("all_finite")) for step in steps),
        "finite_step_counts": {name: sum(bool(step.get(name)) for step in steps) for name in finite_names},
        "metrics": metrics,
    }


def _numeric_model_config(config: NumericTrendConfig) -> ModelConfig:
    spec = MODEL_SPECS[config.model_size]
    return ModelConfig(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        batch_size=config.batch_size,
        d_model=spec.d_model,
        d_ff=spec.d_ff,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
    )


def _cuda_bf16_device(requested: torch.device) -> torch.device:
    if requested.type != "cuda":
        raise ValueError("numeric-trend requires a CUDA device because it compares CUDA BF16 autocast with FP32.")
    if not torch.cuda.is_available():
        raise ValueError("numeric-trend requires CUDA, but CUDA is not available in this PyTorch environment.")
    if requested.index is not None and requested.index >= torch.cuda.device_count():
        raise ValueError(f"Requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) are available.")
    device = torch.device("cuda", torch.cuda.current_device()) if requested.index is None else requested
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError(f"{device} does not support CUDA BF16 autocast.")
    return device


def _new_optimizer(model: nn.Module, config: NumericTrendConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.lr_max,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def _forward_loss_for_training(
    *,
    model: nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    use_bf16_autocast: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16_autocast else nullcontext()
    with autocast:
        logits = model(x)
        loss = cross_entropy(logits, y)
    return logits, loss


def _finish_train_step(*, model: nn.Module, optimizer: AdamW, loss: torch.Tensor, global_step: int, config: NumericTrendConfig) -> None:
    """Apply exactly the benchmark's train-step update after an observation."""

    loss.backward()
    clip_gradient(model.parameters(), config.grad_clip)
    learning_rate = get_cosine_lr(
        global_step + 1,
        config.lr_max,
        config.lr_min,
        config.warmup_steps,
        config.warmup_steps + config.measurement_steps,
    )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    optimizer.step()


def _paired_train_step(
    *,
    fp32_model: nn.Module,
    bf16_model: nn.Module,
    fp32_optimizer: AdamW,
    bf16_optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    global_step: int,
    config: NumericTrendConfig,
    observe: bool,
) -> dict[str, float | bool | None] | None:
    """Run both independent optimizers from comparable state and optionally observe before updating."""

    fp32_logits, fp32_loss = _forward_loss_for_training(
        model=fp32_model,
        optimizer=fp32_optimizer,
        x=x,
        y=y,
        use_bf16_autocast=False,
    )
    bf16_logits, bf16_loss = _forward_loss_for_training(
        model=bf16_model,
        optimizer=bf16_optimizer,
        x=x,
        y=y,
        use_bf16_autocast=True,
    )
    metrics = numeric_error_metrics(
        fp32_logits=fp32_logits,
        bf16_logits=bf16_logits,
        fp32_loss=fp32_loss,
        bf16_loss=bf16_loss,
    ) if observe else None
    _finish_train_step(model=fp32_model, optimizer=fp32_optimizer, loss=fp32_loss, global_step=global_step, config=config)
    _finish_train_step(model=bf16_model, optimizer=bf16_optimizer, loss=bf16_loss, global_step=global_step, config=config)
    return metrics


def numeric_trend_experiment(*, device: torch.device, config: NumericTrendConfig = NumericTrendConfig()) -> dict[str, Any]:
    """Measure ten pre-update numerical comparisons with shared initial weights and inputs."""

    device = _cuda_bf16_device(device)
    model_config = _numeric_model_config(config)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # The reference state is created once on CPU, then copied independently into
    # the FP32 and BF16-autocast models.  It never shares mutable storage with either.
    reference_model = build_model(model_config)
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in reference_model.state_dict().items()}
    fp32_model = build_model(model_config).to(device)
    bf16_model = build_model(model_config).to(device)
    fp32_model.load_state_dict(initial_state)
    bf16_model.load_state_dict(initial_state)
    del reference_model

    input_generator = torch.Generator(device="cpu")
    input_generator.manual_seed(config.seed)
    shape = (model_config.batch_size, model_config.context_length)
    x = torch.randint(model_config.vocab_size, shape, generator=input_generator, dtype=torch.long).to(device)
    y = torch.randint(model_config.vocab_size, shape, generator=input_generator, dtype=torch.long).to(device)
    fp32_optimizer = _new_optimizer(fp32_model, config)
    bf16_optimizer = _new_optimizer(bf16_model, config)

    for global_step in range(config.warmup_steps):
        _paired_train_step(
            fp32_model=fp32_model,
            bf16_model=bf16_model,
            fp32_optimizer=fp32_optimizer,
            bf16_optimizer=bf16_optimizer,
            x=x,
            y=y,
            global_step=global_step,
            config=config,
            observe=False,
        )
        torch.cuda.synchronize(device)

    steps: list[dict[str, Any]] = []
    for measurement_index in range(config.measurement_steps):
        global_step = config.warmup_steps + measurement_index
        metrics = _paired_train_step(
            fp32_model=fp32_model,
            bf16_model=bf16_model,
            fp32_optimizer=fp32_optimizer,
            bf16_optimizer=bf16_optimizer,
            x=x,
            y=y,
            global_step=global_step,
            config=config,
            observe=True,
        )
        assert metrics is not None
        steps.append({"measurement_step": measurement_index + 1, "global_step": global_step, **metrics})
        torch.cuda.synchronize(device)

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "configuration": {
            **asdict(config),
            "mode": "train_step",
            "fp32_precision": "fp32",
            "bf16_precision": "bf16_autocast",
            "model_config": asdict(model_config),
        },
        "comparison": {
            "initialization": "shared_cpu_fp32_state_dict",
            "inputs": "shared_fixed_token_and_target_tensors",
            "optimizer": "independent_matching_adamw_optimizers",
            "observation_point": "before_optimizer_update",
        },
        "environment": {
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
        },
        "steps": steps,
        "summary": summarize_numeric_steps(steps),
    }


def accumulation_experiment() -> dict[str, dict[str, float | str]]:
    """Run the four accumulation variants from the handout without rewriting them."""

    variants: tuple[tuple[str, torch.dtype, torch.dtype, bool], ...] = (
        ("fp32_accumulator_fp32_input", torch.float32, torch.float32, False),
        ("fp16_accumulator_fp16_input", torch.float16, torch.float16, False),
        ("fp32_accumulator_fp16_input", torch.float32, torch.float16, False),
        ("fp32_accumulator_explicit_cast_fp16_input", torch.float32, torch.float16, True),
    )
    results: dict[str, dict[str, float | str]] = {}
    for name, accumulator_dtype, input_dtype, explicit_cast in variants:
        total = torch.tensor(0, dtype=accumulator_dtype)
        for _ in range(1000):
            increment = torch.tensor(0.01, dtype=input_dtype)
            if explicit_cast:
                increment = increment.type(torch.float32)
            total += increment
        value = float(total)
        results[name] = {
            "accumulator_dtype": str(accumulator_dtype),
            "input_dtype": str(input_dtype),
            "value": value,
            "absolute_error_from_10": abs(value - 10.0),
        }
    return results


class ToyModel(nn.Module):
    """The exact small model supplied in the mixed-precision handout."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def toy_dtype_experiment(*, autocast_dtype: torch.dtype, device: torch.device) -> dict[str, str]:
    """Measure actual dtypes instead of assuming an autocast policy."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("ToyModel autocast experiment requires CUDA.")
    model = ToyModel(in_features=16, out_features=4).to(device)
    inputs = torch.randn(8, 16, device=device)
    targets = torch.randint(4, (8,), device=device)
    observed: dict[str, str] = {"parameters": str(next(model.parameters()).dtype)}

    def capture(name: str):
        def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed[name] = str(output.dtype)

        return hook

    hooks = [model.fc1.register_forward_hook(capture("fc1_output")), model.ln.register_forward_hook(capture("layer_norm_output"))]
    try:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(inputs)
            loss = functional.cross_entropy(logits, targets)
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()

    observed["logits"] = str(logits.dtype)
    observed["loss"] = str(loss.dtype)
    observed["gradient"] = str(model.fc1.weight.grad.dtype)
    return observed


def load_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_output(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Task 3 diagnostic mixed-precision experiments.")
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    for name in ("accumulation", "toy", "numeric-trend"):
        command = subparsers.add_parser(name)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    toy = subparsers.choices["toy"]
    toy.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    toy.add_argument("--device", default="cuda")
    numeric_trend = subparsers.choices["numeric-trend"]
    numeric_trend.add_argument("--device", default="cuda", help="CUDA device for the fixed FP32-versus-BF16 diagnostic.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = load_output(args.output)
    output["timestamp_utc"] = datetime.now(UTC).isoformat()
    if args.experiment == "accumulation":
        output["accumulation"] = accumulation_experiment()
    elif args.experiment == "toy":
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        output[f"toy_{args.dtype}"] = toy_dtype_experiment(autocast_dtype=dtype, device=torch.device(args.device))
    else:
        output["language_model_numeric_trend"] = numeric_trend_experiment(device=torch.device(args.device))
    save_output(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    main()
