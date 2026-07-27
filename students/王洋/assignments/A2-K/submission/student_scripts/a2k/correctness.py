from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_systems.a2k.attention import attention_with_lse
from cs336_systems.a2k.flash import FlashAttentionPyTorch, FlashAttentionTriton
from student_scripts.a2k.common import (
    command_string,
    configure_formal_process,
    max_errors,
    public_environment,
    write_json,
)


def saved_lse(output: torch.Tensor, expected_shape: tuple[int, int]) -> torch.Tensor:
    candidates = [tensor for tensor in output.grad_fn.saved_tensors if tensor.shape == expected_shape]
    if len(candidates) != 1:
        raise AssertionError(f"expected one saved LSE tensor, got {len(candidates)}")
    return candidates[0]


def check_one(
    implementation_name: str,
    implementation: type[torch.autograd.Function],
    seed: int,
    head_dimension: int,
    is_causal: bool,
    dtype: torch.dtype,
) -> dict:
    torch.manual_seed(seed)
    batch_size = 2
    sequence_length = 128
    query = torch.randn(batch_size, sequence_length, head_dimension, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    output_gradient = torch.randn_like(query)

    reference_query = query.detach().clone().requires_grad_(True)
    reference_key = key.detach().clone().requires_grad_(True)
    reference_value = value.detach().clone().requires_grad_(True)
    reference_output, reference_lse = attention_with_lse(
        reference_query,
        reference_key,
        reference_value,
        is_causal,
    )
    reference_output.backward(output_gradient)

    output = implementation.apply(query, key, value, is_causal)
    lse = saved_lse(output, (batch_size, sequence_length))
    output.backward(output_gradient)

    tolerances = (2e-2, 2e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
    components = {
        "output": max_errors(output, reference_output),
        "lse": max_errors(lse, reference_lse),
        "d_query": max_errors(query.grad, reference_query.grad),
        "d_key": max_errors(key.grad, reference_key.grad),
        "d_value": max_errors(value.grad, reference_value.grad),
    }
    passed = all(abs_error <= tolerances[1] or rel_error <= tolerances[0] for abs_error, rel_error in components.values())
    return {
        "implementation": implementation_name,
        "seed": seed,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "head_dimension": head_dimension,
        "dtype": str(dtype),
        "causal": is_causal,
        "rtol": tolerances[0],
        "atol": tolerances[1],
        "components": {name: {"max_abs_error": values[0], "max_rel_error": values[1]} for name, values in components.items()},
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended A2-K FlashAttention correctness checks.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fraction = configure_formal_process()
    environment = public_environment(fraction)
    rows = []
    implementations = [("pytorch_tiled", FlashAttentionPyTorch), ("triton", FlashAttentionTriton)]
    for implementation_name, implementation in implementations:
        for seed in (2026, 2027, 2028):
            for head_dimension in (32, 64, 128):
                for is_causal in (False, True):
                    dtype = torch.float32 if seed == 2026 else torch.bfloat16
                    rows.append(check_one(implementation_name, implementation, seed, head_dimension, is_causal, dtype))
    payload = {
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "checks": rows,
        "summary": {"total": len(rows), "passed": sum(row["passed"] for row in rows)},
        "command": command_string(),
        "environment": environment,
    }
    write_json(args.output, payload)
    print(payload["summary"], payload["status"])
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
