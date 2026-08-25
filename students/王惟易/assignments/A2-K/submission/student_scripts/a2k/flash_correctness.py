from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from cs336_systems.a2k.flash_attention_pytorch import FlashAttentionPyTorchFunction
from cs336_systems.a2k.flash_attention_triton import FlashAttentionTritonFunction
from runtime import MINIMUM_FREE_MIB, configure_single_gpu_allocator, synchronize


AttentionFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]
BATCH_SIZE = 2
SEQUENCE_LENGTH = 127
SEEDS = (0, 1, 2)
HEAD_DIMS = (32, 64, 128)
CAUSAL_SETTINGS = (False, True)
IMPLEMENTATIONS: dict[str, AttentionFunction] = {
    "pytorch_tiled": FlashAttentionPyTorchFunction.apply,
    "triton": FlashAttentionTritonFunction.apply,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the extended A2-K FlashAttention correctness matrix.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--minimum-free-mib", type=float, default=MINIMUM_FREE_MIB)
    return parser.parse_args()


def correctness_configs() -> Iterator[tuple[int, int, bool, torch.dtype]]:
    for seed in SEEDS:
        for head_dim in HEAD_DIMS:
            for is_causal in CAUSAL_SETTINGS:
                yield seed, head_dim, is_causal, torch.bfloat16

    # The formal matrix is BF16, but the assignment also requires an FP32
    # correctness configuration.
    yield 0, 64, False, torch.float32


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if is_causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)
        key_positions = torch.arange(k.shape[-2], device=k.device)
        mask = query_positions[:, None] >= key_positions[None, :]
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ v, torch.logsumexp(scores, dim=-1)


def extract_logsumexp(output: torch.Tensor) -> torch.Tensor:
    expected_shape = output.shape[:-1]
    candidates = [
        tensor
        for tensor in output.grad_fn.saved_tensors
        if tensor.shape == expected_shape
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one saved LSE tensor with shape {expected_shape}, found {len(candidates)}")
    return candidates[0]


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-2, 1e-2


def tensor_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> dict[str, float | bool | None]:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    all_finite = bool(torch.isfinite(actual_float).all().item())
    close = all_finite and bool(torch.allclose(actual_float, expected_float, rtol=rtol, atol=atol))
    if not all_finite:
        return {
            "max_absolute_error": None,
            "max_relative_error": None,
            "relative_error_floor": atol,
            "all_finite": False,
            "passed": False,
        }

    absolute_error = (actual_float - expected_float).abs()
    relative_error = absolute_error / expected_float.abs().clamp_min(atol)
    return {
        "max_absolute_error": absolute_error.max().item(),
        "max_relative_error": relative_error.max().item(),
        "relative_error_floor": atol,
        "all_finite": True,
        "passed": close,
    }


def make_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool,
) -> dict[str, torch.Tensor]:
    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    output_ref, logsumexp_ref = reference_attention(q_ref, k_ref, v_ref, is_causal)
    output_ref.backward(grad_output.float())
    return {
        "output": output_ref.detach(),
        "logsumexp": logsumexp_ref.detach(),
        "grad_q": q_ref.grad.detach(),
        "grad_k": k_ref.grad.detach(),
        "grad_v": v_ref.grad.detach(),
    }


def run_implementation(
    name: str,
    implementation: AttentionFunction,
    q_source: torch.Tensor,
    k_source: torch.Tensor,
    v_source: torch.Tensor,
    grad_output: torch.Tensor,
    reference: dict[str, torch.Tensor],
    is_causal: bool,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    q = q_source.detach().clone().requires_grad_(True)
    k = k_source.detach().clone().requires_grad_(True)
    v = v_source.detach().clone().requires_grad_(True)

    output = implementation(q, k, v, is_causal)
    logsumexp = extract_logsumexp(output)
    output.backward(grad_output)
    synchronize()

    metrics = {
        "output": tensor_metrics(output, reference["output"], rtol, atol),
        "logsumexp": tensor_metrics(logsumexp, reference["logsumexp"], rtol, atol),
        "grad_q": tensor_metrics(q.grad, reference["grad_q"], rtol, atol),
        "grad_k": tensor_metrics(k.grad, reference["grad_k"], rtol, atol),
        "grad_v": tensor_metrics(v.grad, reference["grad_v"], rtol, atol),
    }
    passed = all(metric["passed"] for metric in metrics.values())
    return {
        "implementation": name,
        "rtol": rtol,
        "atol": atol,
        "metrics": metrics,
        "passed": passed,
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    allocator, environment = configure_single_gpu_allocator(args.minimum_free_mib)

    cases: list[dict[str, Any]] = []
    for seed, head_dim, is_causal, dtype in correctness_configs():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        shape = (BATCH_SIZE, SEQUENCE_LENGTH, head_dim)
        q_source = torch.randn(shape, device="cuda", dtype=dtype)
        k_source = torch.randn(shape, device="cuda", dtype=dtype)
        v_source = torch.randn(shape, device="cuda", dtype=dtype)
        grad_output = torch.randn(shape, device="cuda", dtype=dtype)
        reference = make_reference(
            q_source,
            k_source,
            v_source,
            grad_output,
            is_causal,
        )
        rtol, atol = tolerances(dtype)

        for name, implementation in IMPLEMENTATIONS.items():
            implementation_result = run_implementation(
                name,
                implementation,
                q_source,
                k_source,
                v_source,
                grad_output,
                reference,
                is_causal,
                rtol,
                atol,
            )
            cases.append(
                {
                    "seed": seed,
                    "shape": list(shape),
                    "dtype": str(dtype).removeprefix("torch."),
                    "is_causal": is_causal,
                    **implementation_result,
                }
            )

        del q_source, k_source, v_source, grad_output, reference
        torch.cuda.empty_cache()

    failed_cases = [case for case in cases if not case["passed"]]
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": f"uv run python {shlex.join(sys.argv)}",
        "environment": environment,
        "allocator": allocator,
        "config": {
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "seeds": list(SEEDS),
            "head_dims": list(HEAD_DIMS),
            "causal_settings": list(CAUSAL_SETTINGS),
            "implementations": list(IMPLEMENTATIONS),
            "reference_dtype": "fp32",
        },
        "summary": {
            "total_cases": len(cases),
            "passed_cases": len(cases) - len(failed_cases),
            "failed_cases": len(failed_cases),
            "status": "ok" if not failed_cases else "failed",
        },
        "cases": cases,
    }
    write_result(args.output, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote correctness results to {args.output}")
    if failed_cases:
        raise AssertionError(f"{len(failed_cases)} FlashAttention correctness cases failed")


if __name__ == "__main__":
    main()
