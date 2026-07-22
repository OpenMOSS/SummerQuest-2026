"""Benchmark one eager or compiled attention/model configuration."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from student_scripts.a2k.common import make_metadata, measure, run_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("attention", "model"), required=True)
    parser.add_argument(
        "--implementation",
        choices=("eager", "compiled"),
        required=True,
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--phase",
        choices=("forward", "backward", "forward_backward", "train_step"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "attention" and args.phase == "train_step":
        parser.error("attention compile comparison does not define train_step")
    if args.kind == "model" and args.phase == "backward":
        parser.error("the model comparison requires forward, forward_backward, or train_step")
    return args


def _cold_time(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _attention_body(args: argparse.Namespace) -> dict[str, object]:
    dtype = torch.bfloat16
    q = torch.randn(
        (1, args.sequence_length, args.head_dim),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    d_output = torch.randn_like(q)

    def eager_forward() -> torch.Tensor:
        return explicit_attention(q, k, v, True)

    forward: Callable[[], torch.Tensor]
    if args.implementation == "compiled":
        forward = torch.compile(
            eager_forward,
            backend="inductor",
            dynamic=False,
        )
    else:
        forward = eager_forward

    if args.phase == "forward":
        measured = forward
        reset_grads = None
    elif args.phase == "backward":
        output: torch.Tensor | None = None

        def prepare_backward_graph() -> None:
            nonlocal output
            output = forward()
            torch.cuda.synchronize()

        prepare_backward_graph()

        def measured() -> None:
            assert output is not None
            output.backward(d_output, retain_graph=True)

        reset_grads = (q, k, v)
    else:

        def measured() -> None:
            output = forward()
            output.backward(d_output)

        reset_grads = (q, k, v)

    cold_start_ms = None
    if args.implementation == "compiled":
        for tensor in (q, k, v):
            tensor.grad = None
        if args.phase == "backward":
            cold_start_ms = _cold_time(measured)
            for tensor in (q, k, v):
                tensor.grad = None
            prepare_backward_graph()
        else:
            cold_start_ms = _cold_time(measured)
        for tensor in (q, k, v):
            tensor.grad = None

    return {
        "kind": "attention",
        "implementation": args.implementation,
        "sequence_length": args.sequence_length,
        "head_dim": args.head_dim,
        "batch_size": 1,
        "dtype": "bf16",
        "causal": True,
        "phase": args.phase,
        "compile_backend": (
            "inductor_fullgraph_static"
            if args.implementation == "compiled"
            else None
        ),
        "cold_start_ms": cold_start_ms,
        **measure(
            measured,
            warmup_ms=100,
            rep_ms=300,
            reset_grads=reset_grads,
        ),
    }


def _model_body(args: argparse.Namespace) -> dict[str, object]:
    from cs336_basics.model import BasicsTransformerLM

    dtype = torch.bfloat16
    config = {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    }
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=args.sequence_length,
        **config,
    ).to("cuda")
    tokens = torch.randint(
        0,
        10_000,
        (1, args.sequence_length),
        device="cuda",
    )
    targets = torch.randint(
        0,
        10_000,
        (1, args.sequence_length),
        device="cuda",
    )
    parameters = tuple(model.parameters())
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=1e-4)
        if args.phase == "train_step"
        else None
    )
    if optimizer is not None:
        with torch.autocast(device_type="cuda", dtype=dtype):
            eager_logits = model(tokens)
            eager_loss = torch.nn.functional.cross_entropy(
                eager_logits.float().reshape(-1, eager_logits.shape[-1]),
                targets.reshape(-1),
            )
        eager_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    executable: Callable[[torch.Tensor], torch.Tensor]
    if args.implementation == "compiled":
        executable = torch.compile(
            model,
            backend="inductor",
            dynamic=False,
        )
    else:
        executable = model

    def forward_loss() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = executable(tokens)
            return torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

    if args.phase == "forward":

        def measured() -> None:
            with torch.autocast(device_type="cuda", dtype=dtype):
                executable(tokens)

        reset_grads = None
    elif args.phase == "forward_backward":

        def measured() -> None:
            forward_loss().backward()

        reset_grads = parameters
    else:
        assert optimizer is not None

        def measured() -> None:
            optimizer.zero_grad(set_to_none=True)
            forward_loss().backward()
            optimizer.step()

        reset_grads = None

    cold_start_ms = None
    if args.implementation == "compiled":
        for parameter in parameters:
            parameter.grad = None
        cold_start_ms = _cold_time(measured)
        for parameter in parameters:
            parameter.grad = None

    return {
        "kind": "model",
        "implementation": args.implementation,
        "model_size": "small",
        "num_layers": config["num_layers"],
        "sequence_length": args.sequence_length,
        "batch_size": 1,
        "dtype": "bf16_autocast_fp32_params",
        "phase": args.phase,
        "compile_backend": (
            "inductor_fullgraph_static_model"
            if args.implementation == "compiled"
            else None
        ),
        "cold_start_ms": cold_start_ms,
        **measure(
            measured,
            warmup_ms=100,
            rep_ms=300,
            reset_grads=reset_grads,
        ),
    }


def main() -> int:
    args = parse_args()
    metadata = make_metadata(args.seed, " ".join(__import__("sys").argv))
    metadata["experiment"] = {
        "kind": args.kind,
        "implementation": args.implementation,
        "sequence_length": args.sequence_length,
        "head_dim": args.head_dim if args.kind == "attention" else None,
        "phase": args.phase,
    }

    def body() -> dict[str, object]:
        if args.kind == "attention":
            return _attention_body(args)
        return _model_body(args)

    return run_safe(args.output, metadata, body)


if __name__ == "__main__":
    raise SystemExit(main())
