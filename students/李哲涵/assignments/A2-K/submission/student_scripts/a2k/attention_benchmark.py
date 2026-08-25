from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from typing import Any

import torch

from cs336_systems.a2k.attention import (
    get_flashattention_autograd_function_triton,
    get_triton_forward_config,
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

PHASES = ("forward", "backward", "forward_backward")
IMPLEMENTATIONS = ("torch_eager", "torch_compiled", "flash_triton")


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        positions = torch.arange(q.shape[-2], device=q.device)
        mask = positions[:, None] >= positions[None, :]
        scores = torch.where(mask, scores, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def build_backend(implementation: str) -> Callable:
    if implementation == "torch_eager":
        return explicit_attention
    if implementation == "torch_compiled":
        return torch.compile(explicit_attention)
    if implementation == "flash_triton":
        flash = get_flashattention_autograd_function_triton()
        return lambda q, k, v, is_causal: flash.apply(q, k, v, is_causal)
    raise ValueError(f"Unknown implementation: {implementation}")


def make_inputs(
    seq_len: int,
    head_dim: int,
    phase: str,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    requires_grad = phase != "forward"
    shape = (1, seq_len, head_dim)
    q = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    dout = torch.randn(shape, dtype=torch.bfloat16, device=device)
    return q, k, v, dout


def clear_gradients(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def measure_cold_start(
    backend: Callable,
    phase: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    causal: bool,
    device: torch.device,
) -> float | None:
    if phase == "backward":
        retained = backend(q, k, v, causal)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        retained.backward(dout, retain_graph=True)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000
        clear_gradients(q, k, v)
        return elapsed

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = backend(q, k, v, causal)
    if phase == "forward_backward":
        output.backward(dout)
        clear_gradients(q, k, v)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000


def do_bench(
    backend: Callable,
    phase: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    causal: bool,
) -> tuple[float, float, float]:
    import triton.testing

    if phase == "forward":
        def fn():
            return backend(q, k, v, causal)

        grad_to_none = None
    elif phase == "backward":
        retained = backend(q, k, v, causal)

        def fn():
            return retained.backward(dout, retain_graph=True)

        grad_to_none = [q, k, v]
    else:
        def fn():
            return backend(q, k, v, causal).backward(dout)

        grad_to_none = [q, k, v]

    quantiles = triton.testing.do_bench(
        fn,
        warmup=100,
        rep=300,
        quantiles=[0.2, 0.5, 0.8],
        grad_to_none=grad_to_none,
    )
    return tuple(float(value) for value in quantiles)


def measure_memory(
    backend: Callable,
    phase: str,
    seq_len: int,
    head_dim: int,
    causal: bool,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    q, k, v, dout = make_inputs(seq_len, head_dim, phase, device, seed + 10_000)
    if phase == "backward":
        output = backend(q, k, v, causal)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        output.backward(dout)
    else:
        torch.cuda.reset_peak_memory_stats(device)
        output = backend(q, k, v, causal)
        if phase == "forward_backward":
            output.backward(dout)
    return peak_memory(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device, metadata = configure_formal_cuda(
        require_4090=not args.allow_nonstandard_gpu,
        min_free_mib=0 if args.allow_low_free_memory else 22 * 1024,
    )
    backend = build_backend(args.implementation)
    q, k, v, dout = make_inputs(
        args.seq_len,
        args.head_dim,
        args.phase,
        device,
        args.seed,
    )

    cold_start_ms = measure_cold_start(
        backend,
        args.phase,
        q,
        k,
        v,
        dout,
        True,
        device,
    )
    p20_ms, p50_ms, p80_ms = do_bench(
        backend,
        args.phase,
        q,
        k,
        v,
        dout,
        True,
    )
    memory = measure_memory(
        backend,
        args.phase,
        args.seq_len,
        args.head_dim,
        True,
        device,
        args.seed,
    )

    launch = (
        get_triton_forward_config(torch.bfloat16, args.head_dim)
        if args.implementation == "flash_triton"
        else {
            "block_q": None,
            "block_k": None,
            "num_warps": None,
            "num_stages": None,
        }
    )
    return {
        "status": "success",
        "implementation": args.implementation,
        "batch_size": 1,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "dtype": "bf16",
        "causal": True,
        "phase": args.phase,
        "warmup_ms": 100,
        "measurement_ms": 300,
        "quantiles": [0.2, 0.5, 0.8],
        "cold_start_ms": cold_start_ms
        if args.implementation == "torch_compiled"
        else None,
        "p20_ms": p20_ms,
        "p50_ms": p50_ms,
        "p80_ms": p80_ms,
        **memory,
        **launch,
        "within_24gib": memory["peak_reserved_mib"] <= 23 * 1024,
        "seed": args.seed,
        "commit": current_commit(),
        "metadata": metadata,
        "command": public_command(
            "student_scripts/a2k/attention_benchmark.py",
            [
                "--implementation",
                args.implementation,
                "--seq-len",
                str(args.seq_len),
                "--head-dim",
                str(args.head_dim),
                "--phase",
                args.phase,
            ],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--head-dim", type=int, choices=[32, 64, 128], required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-nonstandard-gpu", action="store_true")
    parser.add_argument("--allow-low-free-memory", action="store_true")
    args = parser.parse_args()

    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "status": "oom",
            "implementation": args.implementation,
            "batch_size": 1,
            "seq_len": args.seq_len,
            "head_dim": args.head_dim,
            "dtype": "bf16",
            "causal": True,
            "phase": args.phase,
            "warmup_ms": 100,
            "measurement_ms": 300,
            "quantiles": [0.2, 0.5, 0.8],
            "commit": current_commit(),
            "metadata": best_effort_formal_metadata(),
            "command": public_command(
                "student_scripts/a2k/attention_benchmark.py",
                [
                    "--implementation",
                    args.implementation,
                    "--seq-len",
                    str(args.seq_len),
                    "--head-dim",
                    str(args.head_dim),
                    "--phase",
                    args.phase,
                ],
            ),
            **exception_payload(exc),
        }
        if torch.cuda.is_available():
            result.update(peak_memory(torch.device("cuda", 0)))
    except Exception as exc:
        result = {
            "status": "error",
            "implementation": args.implementation,
            "batch_size": 1,
            "seq_len": args.seq_len,
            "head_dim": args.head_dim,
            "dtype": "bf16",
            "causal": True,
            "phase": args.phase,
            "warmup_ms": 100,
            "measurement_ms": 300,
            "quantiles": [0.2, 0.5, 0.8],
            "commit": current_commit(),
            "metadata": best_effort_formal_metadata(),
            "command": public_command(
                "student_scripts/a2k/attention_benchmark.py",
                [
                    "--implementation",
                    args.implementation,
                    "--seq-len",
                    str(args.seq_len),
                    "--head-dim",
                    str(args.head_dim),
                    "--phase",
                    args.phase,
                ],
            ),
            **exception_payload(exc),
        }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
