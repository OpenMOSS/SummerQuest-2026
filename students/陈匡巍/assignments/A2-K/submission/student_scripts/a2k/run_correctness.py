"""Extended output, LSE, and gradient checks for both autograd paths."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch

from cs336_systems.a2k.attention import (
    FlashAttentionPyTorch,
    FlashAttentionTriton,
)
from student_scripts.a2k.common import (
    configure_single_gpu,
    public_gpu_metadata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if is_causal:
        indices = torch.arange(q.shape[-2], device=q.device)
        scores = scores.masked_fill(indices[None, :] > indices[:, None], float("-inf"))
    return torch.matmul(torch.softmax(scores, -1), v), torch.logsumexp(scores, -1)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-3)
    return difference.max().item(), relative.max().item()


def main() -> int:
    args = parse_args()
    allocator = configure_single_gpu()
    gpu = public_gpu_metadata()
    torch.backends.cuda.matmul.allow_tf32 = False
    records: list[dict] = []
    implementations: list[tuple[str, Callable]] = [
        ("pytorch_tiled", FlashAttentionPyTorch.apply),
        ("triton", FlashAttentionTriton.apply),
    ]

    for seed in range(3):
        dtype = torch.float32 if seed == 0 else torch.bfloat16
        tolerance = 0.02 if dtype == torch.float32 else 0.06
        for head_dim in (32, 64, 128):
            for is_causal in (False, True):
                torch.manual_seed(seed)
                q_base = torch.randn(1, 128, head_dim, device="cuda", dtype=dtype)
                k_base = torch.randn_like(q_base)
                v_base = torch.randn_like(q_base)
                grad_output = torch.randn_like(q_base)

                q_reference = q_base.detach().clone().requires_grad_(True)
                k_reference = k_base.detach().clone().requires_grad_(True)
                v_reference = v_base.detach().clone().requires_grad_(True)
                expected_output, expected_lse = reference(q_reference, k_reference, v_reference, is_causal)
                expected_grads = torch.autograd.grad(
                    expected_output,
                    (q_reference, k_reference, v_reference),
                    grad_output,
                )

                for implementation_name, implementation in implementations:
                    q = q_base.detach().clone().requires_grad_(True)
                    k = k_base.detach().clone().requires_grad_(True)
                    v = v_base.detach().clone().requires_grad_(True)
                    output = implementation(q, k, v, is_causal)
                    saved_lse = [tensor for tensor in output.grad_fn.saved_tensors if tensor.shape == (1, 128)][0]
                    actual_grads = torch.autograd.grad(output, (q, k, v), grad_output)

                    components = {
                        "output": (output, expected_output),
                        "logsumexp": (saved_lse, expected_lse),
                        "dQ": (actual_grads[0], expected_grads[0]),
                        "dK": (actual_grads[1], expected_grads[1]),
                        "dV": (actual_grads[2], expected_grads[2]),
                    }
                    metrics = {}
                    passed = True
                    for name, (actual, expected) in components.items():
                        absolute, relative = error_metrics(actual, expected)
                        metrics[name] = {
                            "max_absolute_error": absolute,
                            "max_relative_error": relative,
                        }
                        passed &= torch.allclose(
                            actual.float(),
                            expected.float(),
                            atol=tolerance,
                            rtol=tolerance,
                        )

                    records.append(
                        {
                            "implementation": implementation_name,
                            "seed": seed,
                            "shape": [1, 128, head_dim],
                            "dtype": str(dtype).removeprefix("torch."),
                            "is_causal": is_causal,
                            "absolute_tolerance": tolerance,
                            "relative_tolerance": tolerance,
                            "metrics": metrics,
                            "status": "pass" if passed else "fail",
                        }
                    )

    payload = {
        "summary": {
            "configurations": len(records),
            "passed": sum(row["status"] == "pass" for row in records),
            "failed": sum(row["status"] != "pass" for row in records),
            "tf32_enabled": False,
        },
        "allocator": allocator,
        "gpu": gpu,
        "command": "python -m student_scripts.a2k.run_correctness --output results/correctness.json",
        "records": records,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summary"]))
    return 0 if payload["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
