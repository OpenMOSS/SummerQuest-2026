from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from cs336_systems.a2k.attention import FlashAttentionPyTorch, explicit_attention_with_lse
from cs336_systems.a2k.runtime import configure_cuda_allocator, write_json
from student_scripts.a2k.common import torch_dtype


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="A2-K extended FlashAttention correctness matrix")
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--metadata-output", type=Path, required=True)
    result.add_argument("--allocator-limit-mib", type=int, default=23552)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--sequence-length", type=int, default=128)
    result.add_argument("--dry-run", action="store_true")
    return result.parse_args()


def error_pair(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    # Relative error is not informative for reference values near zero. Use the
    # same scale as the public comparison floor instead of reporting enormous
    # ratios for numerically insignificant gradient entries.
    scale = expected.float().abs().clamp_min(1e-2)
    return float(difference.max().item()), float((difference / scale).max().item())


def _extract_lse(output: torch.Tensor) -> torch.Tensor:
    candidates = [tensor for tensor in output.grad_fn.saved_tensors if tensor.shape == output.shape[:-1]]
    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one saved LSE tensor, found {len(candidates)}")
    return candidates[0]


def _reference_gradients(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_output: torch.Tensor, causal: bool):
    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    output, lse = explicit_attention_with_lse(q_ref, k_ref, v_ref, causal)
    output.backward(grad_output)
    return output.detach(), lse.detach(), q_ref.grad.detach(), k_ref.grad.detach(), v_ref.grad.detach()


def _candidate_gradients(function: type, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_output: torch.Tensor, causal: bool):
    q_candidate, k_candidate, v_candidate = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    output = function.apply(q_candidate, k_candidate, v_candidate, causal)
    lse = _extract_lse(output).detach()
    output.backward(grad_output)
    return output.detach(), lse, q_candidate.grad.detach(), k_candidate.grad.detach(), v_candidate.grad.detach()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parser()
    dimensions = (32, 64, 128)
    dtypes = ("fp32", "bf16")
    causal_values = (False, True)
    combinations = [(seed, dim, dtype, causal) for seed in range(args.seed, args.seed + 3) for dim in dimensions for dtype in dtypes for causal in causal_values]
    if args.dry_run:
        print(f"correctness cases: {len(combinations)}")
        return 0
    if args.device == "cuda":
        allocator = configure_cuda_allocator(allocator_limit_mib=args.allocator_limit_mib)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        metadata = {
            "allocator": allocator.to_dict(),
            "device": torch.cuda.get_device_name(0),
            "tf32_enabled": False,
        }
    else:
        allocator = None
        metadata = {"device": "cpu", "tf32_enabled": False}

    commit = _commit()
    metadata.update(
        {
            "commit": commit,
            "seed_start": args.seed,
            "seed_values": list(range(args.seed, args.seed + 3)),
            "sequence_length": args.sequence_length,
            "head_dims": [32, 64, 128],
            "dtypes": ["fp32", "bf16"],
            "causal_values": [False, True],
            "fields": ["output", "lse", "dq", "dk", "dv"],
            "timer": "CUDA synchronization around each candidate evaluation",
        }
    )

    implementations: list[tuple[str, type]] = [("pytorch", FlashAttentionPyTorch)]
    if args.device == "cuda" and torch.cuda.is_available():
        from cs336_systems.a2k.triton_attention import FlashAttentionTriton

        implementations.append(("triton", FlashAttentionTriton))
    else:
        metadata["triton_status"] = "skip: CUDA is unavailable"

    rows: list[dict[str, object]] = []
    for seed, head_dim, dtype_name, causal in combinations:
        torch.manual_seed(seed)
        dtype = torch_dtype(dtype_name)
        q = torch.randn(1, args.sequence_length, head_dim, device=args.device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad_output = torch.randn_like(q)
        expected = _reference_gradients(q, k, v, grad_output, causal)
        for implementation_name, function in implementations:
            try:
                actual = _candidate_gradients(function, q, k, v, grad_output, causal)
                tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
                for field, actual_tensor, expected_tensor in (
                    ("output", actual[0], expected[0]),
                    ("lse", actual[1], expected[1]),
                    ("dq", actual[2], expected[2]),
                    ("dk", actual[3], expected[3]),
                    ("dv", actual[4], expected[4]),
                ):
                    atol = rtol = tolerance
                    max_abs, max_rel = error_pair(actual_tensor, expected_tensor)
                    passed = bool(torch.allclose(actual_tensor.float(), expected_tensor.float(), atol=atol, rtol=rtol))
                    rows.append({
                        "implementation": implementation_name,
                        "seed": seed,
                        "sequence_length": args.sequence_length,
                        "head_dim": head_dim,
                        "dtype": dtype_name,
                        "is_causal": causal,
                        "field": field,
                        "max_abs_error": max_abs,
                        "max_relative_error": max_rel,
                        "atol": atol,
                        "rtol": rtol,
                        "status": "pass" if passed else "fail",
                    })
            except torch.OutOfMemoryError as error:
                rows.append({"implementation": implementation_name, "seed": seed, "sequence_length": args.sequence_length, "head_dim": head_dim, "dtype": dtype_name, "is_causal": causal, "field": "all", "status": "oom", "error": str(error)[:500]})

    write_json(args.output, {"cases": rows, "summary": {"total": len(rows), "failed": sum(row.get("status") == "fail" for row in rows)}, "metadata": metadata})
    write_json(
        args.metadata_output,
        {
            "experiment": "correctness",
            "commit": commit,
            "metadata": metadata,
            "command": ["python", "-m", "student_scripts.a2k.correctness", *sys.argv[1:]],
        },
    )
    print(f"wrote {len(rows)} correctness rows to {args.output}")
    return 0 if not any(row.get("status") == "fail" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
