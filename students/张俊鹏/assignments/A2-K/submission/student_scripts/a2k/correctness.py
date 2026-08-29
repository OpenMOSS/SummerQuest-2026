from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import triton

from cs336_systems.a2k.flash_attention import (
    FlashAttentionPytorch,
    _tiled_attention_forward,
)
from cs336_systems.a2k.flash_attention_triton import (
    FlashAttentionTriton,
    _flash_attention_forward_kernel,
)


def max_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs_error": float(difference.max().item()),
        "max_rel_error": float(relative.max().item()),
    }


def triton_forward_with_lse(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_queries, head_dim = q.shape
    n_keys = k.shape[1]
    output = torch.empty_like(q)
    lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
    block_m = 32 if head_dim > 64 else 64
    block_n = 32 if head_dim > 64 else 64
    block_d = triton.next_power_of_2(head_dim)
    _flash_attention_forward_kernel[(triton.cdiv(n_queries, block_m), batch_size)](
        q, k, v, output, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        n_queries, n_keys, head_dim, 1.0 / math.sqrt(head_dim),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        IS_CAUSAL=is_causal, num_warps=4,
        num_stages=1 if head_dim > 64 else 2,
    )
    return output, lse


def main() -> None:
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocator_fraction = min(1.0, 23 * 1024**3 / total_memory)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records: list[dict[str, object]] = []
    for seed in (11, 29, 47):
        for head_dim in (32, 64, 128):
            for is_causal in (False, True):
                dtype = torch.float32 if (seed, head_dim, is_causal) == (11, 32, False) else torch.bfloat16
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                shape = (2, 97, head_dim)
                q0 = torch.randn(shape, device="cuda", dtype=dtype)
                k0 = torch.randn_like(q0)
                v0 = torch.randn_like(q0)
                do = torch.randn_like(q0)

                q_ref, k_ref, v_ref = (x.detach().clone().requires_grad_(True) for x in (q0, k0, v0))
                reference_output, reference_lse = _tiled_attention_forward(q_ref, k_ref, v_ref, is_causal)
                reference_output.backward(do)
                reference_grads = (q_ref.grad, k_ref.grad, v_ref.grad)

                q_tri, k_tri, v_tri = (x.detach().clone().requires_grad_(True) for x in (q0, k0, v0))
                triton_output = FlashAttentionTriton.apply(q_tri, k_tri, v_tri, is_causal)
                triton_output.backward(do)
                triton_grads = (q_tri.grad, k_tri.grad, v_tri.grad)
                _, triton_lse = triton_forward_with_lse(q0, k0, v0, is_causal)

                # Match the official 1e-2 absolute criterion for FP32 and
                # allow two BF16 quantization units for the BF16 path.
                tolerance = 1e-2 if dtype == torch.float32 else 2e-2
                metrics = {
                    "output": max_errors(triton_output, reference_output),
                    "lse": max_errors(triton_lse, reference_lse),
                    "dQ": max_errors(triton_grads[0], reference_grads[0]),
                    "dK": max_errors(triton_grads[1], reference_grads[1]),
                    "dV": max_errors(triton_grads[2], reference_grads[2]),
                }
                passed = all(m["max_abs_error"] <= tolerance for m in metrics.values())
                records.append({
                    "seed": seed,
                    "shape": list(shape),
                    "dtype": str(dtype).replace("torch.", ""),
                    "causal": is_causal,
                    "tf32_enabled": False,
                    "tolerance": {"max_abs": tolerance, "max_rel": None},
                    "metrics": metrics,
                    "passed": passed,
                })
    payload = {
        "reference": "tiled PyTorch FlashAttention reference",
        "candidate": "student Triton FlashAttention",
        "allocator_limit_mib": 23552,
        "allocator_fraction": allocator_fraction,
        "tf32_enabled": False,
        "records": records,
        "passed": all(record["passed"] for record in records),
    }
    path = Path("../SummerQuest-2026/students/张俊鹏/assignments/A2-K/results/correctness.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "passed": payload["passed"], "path": str(path)}, ensure_ascii=False))
    for record in records:
        print(record["seed"], record["shape"], record["dtype"], record["causal"], record["passed"], record["metrics"])


if __name__ == "__main__":
    main()
