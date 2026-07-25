"""Shared model, timing, device, and annotation helpers for A2-P."""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def ensure_assignment1_importable() -> None:
    """Prefer the sibling A1 implementation over the staff fallback package."""

    # Search the required sibling A1 checkout before the staff copy inside the
    # A2 starter repository.  The first entry is the canonical SummerQuest
    # layout; the remaining entries make the synced code portable.
    candidates: list[Path] = [
        REPO_ROOT.parent / "assignment1-basics",
        REPO_ROOT.parent / "cs336-basics",
    ]
    for ancestor in REPO_ROOT.parents:
        candidates.extend(
            (
                ancestor / "assignment1-basics",
                ancestor / "cs336-basics",
            )
        )
    candidates.extend((REPO_ROOT / "cs336-basics",))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "cs336_basics").is_dir():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return
    # If neither sibling repository exists, let the normal Python import path
    # resolve an installed ``cs336_basics`` package.


ensure_assignment1_importable()
try:
    _model_module = importlib.import_module("cs336_basics.transformer")
    TransformerLM = _model_module.TransformerLM
    _uses_student_transformer_api = True
except (ImportError, AttributeError):
    _model_module = importlib.import_module("cs336_basics.model")
    TransformerLM = _model_module.BasicsTransformerLM
    _uses_student_transformer_api = False

try:
    _optim_module = importlib.import_module("cs336_basics.optim")
    AdamW = _optim_module.AdamW
    cross_entropy = _optim_module.cross_entropy
except ImportError:
    _optim_module = importlib.import_module("cs336_basics.optimizer")
    AdamW = _optim_module.AdamW
    cross_entropy = getattr(_optim_module, "cross_entropy", None)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower().replace("float", "fp")
    choices = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    try:
        return choices[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}; choose fp32, fp16, or bf16") from error


def model_spec(name: str) -> dict[str, int]:
    try:
        return dict(MODEL_SPECS[name.lower()])
    except KeyError as error:
        raise ValueError(f"unknown model size {name!r}; choose {', '.join(MODEL_SPECS)}") from error


def build_model(
    model_size: str,
    *,
    vocab_size: int,
    context_length: int,
    device: torch.device,
    parameter_dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    spec = model_spec(model_size)
    kwargs: dict[str, Any] = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": spec["d_model"],
        "num_layers": spec["num_layers"],
        "num_heads": spec["num_heads"],
        "d_ff": spec["d_ff"],
        "rope_theta": 10_000.0,
    }
    if _uses_student_transformer_api:
        kwargs.update({"device": device, "dtype": parameter_dtype})
        model = TransformerLM(**kwargs)
    else:
        model = TransformerLM(**kwargs).to(device=device, dtype=parameter_dtype)
    return model


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def make_batch(
    *,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length),
        generator=generator,
        device="cpu",
        dtype=torch.long,
    )
    return batch.to(device=device, non_blocking=device.type == "cuda")


def make_targets(inputs: torch.Tensor, *, vocab_size: int, seed: int) -> torch.Tensor:
    """Create deterministic next-token targets outside the timed region."""

    # A rolled copy keeps the target shape equal to the model output while
    # still representing a next-token language-model objective.  The final
    # target is generated independently so it does not accidentally copy the
    # first token of the same sequence.
    targets = torch.roll(inputs, shifts=-1, dims=-1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    final = torch.randint(
        0,
        vocab_size,
        (inputs.shape[0],),
        generator=generator,
        device="cpu",
        dtype=torch.long,
    ).to(device=inputs.device, non_blocking=inputs.device.type == "cuda")
    targets[..., -1] = final
    return targets


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, dtype: torch.dtype) -> contextlib.AbstractContextManager[Any]:
    if dtype == torch.float32:
        return contextlib.nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def cuda_range(label: str) -> contextlib.AbstractContextManager[Any]:
    """Return a range visible to torch.profiler and, when available, Nsight."""

    @contextlib.contextmanager
    def manager() -> Iterator[None]:
        record = torch.profiler.record_function(label)
        record.__enter__()
        pushed = False
        if torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_push(label)
                pushed = True
            except (RuntimeError, AttributeError):
                pass
        try:
            yield
        finally:
            if pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except (RuntimeError, AttributeError):
                    pass
            record.__exit__(None, None, None)

    return manager()


def patch_attention_annotations() -> None:
    """Replace the A1 attention helper with an equivalent annotated version."""

    try:
        module = importlib.import_module("cs336_basics.transformer")
    except ImportError:
        # The staff fallback in the starter repository exposes a different
        # model module.  It does not have the standalone attention helper that
        # can be safely monkey-patched here; the rest of the profiler remains
        # usable without the optional substage annotations.
        return
    if not hasattr(module, "scaled_dot_product_attention"):
        return
    if getattr(module, "_a2p_attention_patched", False):
        return

    original = module.scaled_dot_product_attention
    softmax_fn = getattr(module, "softmax", torch.softmax)

    def annotated(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = key.shape[-1]
        with cuda_range("attention/scores"):
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        with cuda_range("attention/softmax"):
            probabilities = softmax_fn(scores, dim=-1)
        with cuda_range("attention/value"):
            return torch.matmul(probabilities, value)

    # Keep a reference for debugging and expose a switchable patch without
    # changing the student's Transformer implementation.
    setattr(module, "_a2p_original_attention", original)
    setattr(module, "scaled_dot_product_attention", annotated)
    setattr(module, "_a2p_attention_patched", True)


def loss_for_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the A1 next-token cross-entropy in FP32."""

    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets must match logits' non-class dimensions: "
            f"got logits {tuple(logits.shape)}, targets {tuple(targets.shape)}"
        )
    # A1's implementation is intentionally used when available.  Casting the
    # logits to FP32 keeps the reduction numerically stable under BF16
    # autocast, while the model forward remains inside the autocast region.
    if cross_entropy is not None:
        return cross_entropy(logits.float(), targets)
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


def run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    mode: str,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if mode not in {"forward", "forward_backward", "train_step"}:
        raise ValueError(f"unsupported mode: {mode}")

    if mode != "forward":
        with cuda_range("zero_grad"):
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

    graph_context: contextlib.AbstractContextManager[Any]
    if mode == "forward":
        graph_context = torch.no_grad()
    else:
        graph_context = contextlib.nullcontext()

    with graph_context:
        with autocast_context(device, amp_dtype):
            with cuda_range("forward"):
                logits = model(inputs)
            if mode == "forward":
                return None
            with cuda_range("loss"):
                if targets is None:
                    raise ValueError("targets are required for backward and train_step modes")
                loss = loss_for_logits(logits, targets)

    with cuda_range("backward"):
        loss.backward()
    if mode == "train_step":
        assert optimizer is not None
        with cuda_range("optimizer"):
            optimizer.step()
    # Keep the scalar on-device.  Calling ``.item()`` here would add a host
    # synchronization and contaminate the timed region; callers extract it
    # only after their explicit measurement synchronization.
    return loss.detach()


def json_safe(value: Any) -> Any:
    """Convert common argparse/PyTorch values into JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def namespace_config(namespace: Any) -> dict[str, Any]:
    """Serialize an argparse Namespace without leaking non-JSON objects."""

    values = vars(namespace) if hasattr(namespace, "__dict__") else namespace
    return json_safe(dict(values))


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep output-path arguments out of public metadata."""

    hidden = {"output", "trace_output", "table_output", "snapshot"}
    return {
        key: value
        for key, value in json_safe(config).items()
        if key not in hidden
    }


def sanitized_command(program: str, argv: list[str]) -> str:
    """Render a reproducible command without exposing local output paths."""

    path_flags = {"--output", "--trace-output", "--table-output", "--snapshot"}
    rendered = [program]
    index = 0
    while index < len(argv):
        argument = str(argv[index])
        rendered.append(argument)
        if argument in path_flags and index + 1 < len(argv):
            index += 1
            rendered.append(Path(str(argv[index])).name)
        index += 1
    return " ".join(rendered)


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "python": sys.version.split()[0],
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            }
        )
        # Keep only public, non-identifying GPU metadata.  Failure to query
        # nvidia-smi should not make a benchmark unusable (for example in a
        # CPU-only smoke test or a restricted container).
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if rows:
                fields = [field.strip() for field in rows[0].split(",")]
                if len(fields) >= 6:
                    result["nvidia_smi"] = {
                        "gpu_name": fields[0],
                        "memory_total_mib": fields[1],
                        "memory_free_mib_at_query": fields[2],
                        "driver_version": fields[3],
                        "power_limit_w": fields[4],
                        "pstate": fields[5],
                    }
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def peak_memory_mib(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / 2**20,
        torch.cuda.max_memory_reserved(device) / 2**20,
    )


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def finite_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": float("nan"), "std_ms": float("nan"), "cv": float("nan")}
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean())
    std = float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0
    return {"mean_ms": mean, "std_ms": std, "cv": std / mean if mean else 0.0}
