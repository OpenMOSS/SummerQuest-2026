from __future__ import annotations

import argparse
import math
from typing import Any

import torch

from cs336_systems.a2k.attention import (
    get_flashattention_autograd_function_pytorch,
    get_flashattention_autograd_function_triton,
)
from cs336_systems.a2k.runtime import (
    best_effort_formal_metadata,
    configure_formal_cuda,
    current_commit,
    exception_payload,
    peak_memory,
    public_command,
    write_json,
)


def reference_attention_and_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores = scores / math.sqrt(q.shape[-1])
    if causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)
        key_positions = torch.arange(k.shape[-2], device=q.device)
        mask = query_positions[:, None] >= key_positions[None, :]
        scores = torch.where(mask, scores, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v.float())
    return output, torch.logsumexp(scores, dim=-1)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    absolute = (actual_fp32 - expected_fp32).abs()
    relative = absolute / expected_fp32.abs().clamp_min(1e-6)
    return {
        "max_abs_error": float(absolute.max()),
        "max_rel_error": float(relative.max()),
    }


def check_one(
    implementation: str,
    seed: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    shape = (2, 128, head_dim)
    q_base = torch.randn(shape, dtype=dtype, device=device)
    k_base = torch.randn(shape, dtype=dtype, device=device)
    v_base = torch.randn(shape, dtype=dtype, device=device)
    dout = torch.randn(shape, dtype=dtype, device=device)

    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)
    output_ref, lse_ref = reference_attention_and_lse(
        q_ref,
        k_ref,
        v_ref,
        causal,
    )
    output_ref.backward(dout.float())

    function = (
        get_flashattention_autograd_function_pytorch()
        if implementation == "flash_pytorch"
        else get_flashattention_autograd_function_triton()
    )
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    output = function.apply(q, k, v, causal)
    saved_lse = [
        tensor
        for tensor in output.grad_fn.saved_tensors
        if tensor.shape == (shape[0], shape[1])
    ]
    if len(saved_lse) != 1:
        raise AssertionError(
            f"Expected one saved LSE tensor, found {len(saved_lse)}"
        )
    output.backward(dout)

    atol = 1e-2 if dtype == torch.float32 else 2e-2
    rtol = 1e-2 if dtype == torch.float32 else 2e-2
    tensors = {
        "output": (output, output_ref),
        "lse": (saved_lse[0], lse_ref),
        "dq": (q.grad, q_ref.grad),
        "dk": (k.grad, k_ref.grad),
        "dv": (v.grad, v_ref.grad),
    }
    metrics = {
        name: error_metrics(actual, expected)
        for name, (actual, expected) in tensors.items()
    }
    passed = all(
        torch.allclose(
            actual.float(),
            expected.float(),
            atol=atol,
            rtol=rtol,
        )
        for actual, expected in tensors.values()
    )
    return {
        "implementation": implementation,
        "seed": seed,
        "batch_size": shape[0],
        "sequence_length": shape[1],
        "head_dim": head_dim,
        "causal": causal,
        "dtype": str(dtype).removeprefix("torch."),
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "atol": atol,
        "rtol": rtol,
        "metrics": metrics,
        "passed": passed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device, metadata = configure_formal_cuda(
        require_4090=not args.allow_nonstandard_gpu,
        min_free_mib=0 if args.allow_low_free_memory else 22 * 1024,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata["tf32_matmul_allowed"] = False
    metadata["tf32_cudnn_allowed"] = False

    rows: list[dict[str, Any]] = []
    for implementation in ("flash_pytorch", "flash_triton"):
        for seed in (0, 1, 2):
            for head_dim in (32, 64, 128):
                for causal in (False, True):
                    rows.append(
                        check_one(
                            implementation,
                            seed,
                            head_dim,
                            causal,
                            torch.bfloat16,
                            device,
                        )
                    )
        rows.append(
            check_one(
                implementation,
                0,
                64,
                False,
                torch.float32,
                device,
            )
        )

    return {
        "status": "success" if all(row["passed"] for row in rows) else "failed",
        "commit": current_commit(),
        "metadata": metadata,
        "command": public_command(
            "student_scripts/a2k/correctness.py",
            [],
        ),
        "summary": {
            "rows": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "failed": sum(not row["passed"] for row in rows),
        },
        **peak_memory(device),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-nonstandard-gpu", action="store_true")
    parser.add_argument("--allow-low-free-memory", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "status": "error",
            "commit": current_commit(),
            "metadata": best_effort_formal_metadata(),
            "command": public_command(
                "student_scripts/a2k/correctness.py",
                [],
            ),
            **exception_payload(exc),
        }
    write_json(args.output, result)
    print(result.get("summary", result))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
