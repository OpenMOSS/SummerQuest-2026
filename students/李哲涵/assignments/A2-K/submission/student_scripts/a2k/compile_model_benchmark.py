from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k.runtime import (
    best_effort_formal_metadata,
    configure_formal_cuda,
    current_commit,
    exception_payload,
    latency_summary,
    peak_memory,
    public_command,
    write_json,
)

SMALL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
}


def execute(
    model: torch.nn.Module,
    optimizer: AdamW | None,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
) -> float | None:
    if mode == "train_step":
        optimizer.zero_grad(set_to_none=True)
    elif mode == "forward_backward":
        model.zero_grad(set_to_none=True)

    grad_context = torch.no_grad() if mode == "forward" else nullcontext()
    with grad_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(tokens)
        loss = None
        if mode != "forward":
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
    if mode != "forward":
        loss.backward()
    if mode == "train_step":
        optimizer.step()
    return None if loss is None else float(loss.detach())


def timed_execute(
    model: torch.nn.Module,
    optimizer: AdamW | None,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    device: torch.device,
) -> tuple[float, float | None]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss = execute(model, optimizer, tokens, targets, mode)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000, loss


def run(args: argparse.Namespace) -> dict[str, Any]:
    device, metadata = configure_formal_cuda(
        require_4090=not args.allow_nonstandard_gpu,
        min_free_mib=0 if args.allow_low_free_memory else 22 * 1024,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = BasicsTransformerLM(**SMALL_CONFIG).to(device)
    optimizer = (
        AdamW(model.parameters(), lr=1e-4)
        if args.mode == "train_step"
        else None
    )
    if args.implementation == "torch_compiled":
        model = torch.compile(model)

    tokens = torch.randint(
        0,
        SMALL_CONFIG["vocab_size"],
        (1, SMALL_CONFIG["context_length"]),
        device=device,
    )
    targets = torch.randint(
        0,
        SMALL_CONFIG["vocab_size"],
        (1, SMALL_CONFIG["context_length"]),
        device=device,
    )

    cold_start_ms, _ = timed_execute(
        model,
        optimizer,
        tokens,
        targets,
        args.mode,
        device,
    )
    for _ in range(args.warmup):
        execute(model, optimizer, tokens, targets, args.mode)
        torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        elapsed_ms, loss = timed_execute(
            model,
            optimizer,
            tokens,
            targets,
            args.mode,
            device,
        )
        samples_ms.append(elapsed_ms)
        if loss is not None:
            losses.append(loss)

    torch.cuda.reset_peak_memory_stats(device)
    execute(model, optimizer, tokens, targets, args.mode)
    memory = peak_memory(device)
    summary = latency_summary(samples_ms)
    return {
        "status": "success",
        "scope": "small_model",
        "implementation": args.implementation,
        "model_size": "small",
        "batch_size": 1,
        "context_length": 512,
        "dtype": "bf16_autocast",
        "parameter_dtype": "fp32",
        "mode": args.mode,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "cold_start_ms": cold_start_ms
        if args.implementation == "torch_compiled"
        else None,
        **summary,
        **memory,
        "loss_samples": losses,
        "seed": args.seed,
        "commit": current_commit(),
        "metadata": metadata,
        "command": public_command(
            "student_scripts/a2k/compile_model_benchmark.py",
            [
                "--implementation",
                args.implementation,
                "--mode",
                args.mode,
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
            ],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=["torch_eager", "torch_compiled"],
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "train_step"],
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
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
            "scope": "small_model",
            "implementation": args.implementation,
            "model_size": "small",
            "batch_size": 1,
            "context_length": 512,
            "dtype": "bf16_autocast",
            "mode": args.mode,
            "commit": current_commit(),
            "metadata": best_effort_formal_metadata(),
            "command": public_command(
                "student_scripts/a2k/compile_model_benchmark.py",
                [
                    "--implementation",
                    args.implementation,
                    "--mode",
                    args.mode,
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                ],
            ),
            **exception_payload(exc),
        }
        if torch.cuda.is_available():
            result.update(peak_memory(torch.device("cuda", 0)))
    except Exception as exc:
        result = {
            "status": "error",
            "scope": "small_model",
            "implementation": args.implementation,
            "model_size": "small",
            "batch_size": 1,
            "context_length": 512,
            "dtype": "bf16_autocast",
            "mode": args.mode,
            "commit": current_commit(),
            "metadata": best_effort_formal_metadata(),
            "command": public_command(
                "student_scripts/a2k/compile_model_benchmark.py",
                [
                    "--implementation",
                    args.implementation,
                    "--mode",
                    args.mode,
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                ],
            ),
            **exception_payload(exc),
        }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
