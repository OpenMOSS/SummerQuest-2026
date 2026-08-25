from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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

MEDIUM_CONFIG = {
    "vocab_size": 10_000,
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "d_ff": 4096,
}


def checkpointed_forward(
    model: BasicsTransformerLM,
    tokens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    hidden = model.token_embeddings(tokens)
    for start in range(0, len(model.layers), block_size):
        layers = model.layers[start : start + block_size]

        def run_layers(x, layers=layers):
            for layer in layers:
                x = layer(x)
            return x

        hidden = checkpoint(run_layers, hidden, use_reentrant=False)
    return model.lm_head(model.ln_final(hidden))


def training_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    block_size: int | None,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = (
            model(tokens)
            if block_size is None
            else checkpointed_forward(model, tokens, block_size)
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run(args: argparse.Namespace) -> dict:
    device, metadata = configure_formal_cuda(
        require_4090=not args.allow_nonstandard_gpu,
        min_free_mib=0 if args.allow_low_free_memory else 22 * 1024,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_config = {
        **MEDIUM_CONFIG,
        "context_length": args.context_length,
    }
    model = BasicsTransformerLM(**model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(
        0,
        model_config["vocab_size"],
        (1, args.context_length),
        device=device,
    )
    targets = torch.randint(
        0,
        model_config["vocab_size"],
        (1, args.context_length),
        device=device,
    )

    result = {
        "status": "success",
        "config_id": (
            f"medium_s{args.context_length}_ckpt"
            f"{args.checkpoint_block_size or 'none'}"
        ),
        "model_size": "medium",
        "num_layers": MEDIUM_CONFIG["num_layers"],
        "context_length": args.context_length,
        "batch_size": 1,
        "dtype": "bf16_autocast",
        "parameter_dtype": "fp32",
        "optimizer": "AdamW",
        "checkpoint_block_size": args.checkpoint_block_size,
        "nested": False,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "seed": args.seed,
        "metadata": metadata,
        "commit": current_commit(),
        "command": public_command(
            "student_scripts/a2k/checkpoint_benchmark.py",
            [
                "--context-length",
                str(args.context_length),
                "--checkpoint-block-size",
                str(args.checkpoint_block_size or 0),
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
            ],
        ),
    }

    for _ in range(args.warmup):
        training_step(
            model,
            optimizer,
            tokens,
            targets,
            args.checkpoint_block_size,
        )
        torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    peak_allocated: list[float] = []
    peak_reserved: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss = training_step(
            model,
            optimizer,
            tokens,
            targets,
            args.checkpoint_block_size,
        )
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000)
        memory = peak_memory(device)
        peak_allocated.append(memory["peak_allocated_mib"])
        peak_reserved.append(memory["peak_reserved_mib"])
        losses.append(loss)

    result.update(latency_summary(samples_ms))
    result.update(
        {
            "step_time_ms_samples": samples_ms,
            "step_time_ms_p50": latency_summary(samples_ms)["p50_ms"],
            "peak_allocated_mib_samples": peak_allocated,
            "peak_reserved_mib_samples": peak_reserved,
            "peak_allocated_mib": max(peak_allocated),
            "peak_reserved_mib": max(peak_reserved),
            "loss_samples": losses,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, choices=[1024, 2048], required=True)
    parser.add_argument(
        "--checkpoint-block-size",
        type=int,
        choices=[0, 1, 2, 4, 8],
        default=0,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-nonstandard-gpu", action="store_true")
    parser.add_argument("--allow-low-free-memory", action="store_true")
    args = parser.parse_args()
    args.checkpoint_block_size = args.checkpoint_block_size or None

    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "status": "oom",
            "config_id": (
                f"medium_s{args.context_length}_ckpt"
                f"{args.checkpoint_block_size or 'none'}"
            ),
            "model_size": "medium",
            "num_layers": MEDIUM_CONFIG["num_layers"],
            "context_length": args.context_length,
            "batch_size": 1,
            "dtype": "bf16_autocast",
            "checkpoint_block_size": args.checkpoint_block_size,
            "commit": current_commit(),
            "metadata": best_effort_formal_metadata(),
            "command": public_command(
                "student_scripts/a2k/checkpoint_benchmark.py",
                [
                    "--context-length",
                    str(args.context_length),
                    "--checkpoint-block-size",
                    str(args.checkpoint_block_size or 0),
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
            "config_id": (
                f"medium_s{args.context_length}_ckpt"
                f"{args.checkpoint_block_size or 'none'}"
            ),
            "model_size": "medium",
            "num_layers": MEDIUM_CONFIG["num_layers"],
            "context_length": args.context_length,
            "batch_size": 1,
            "dtype": "bf16_autocast",
            "checkpoint_block_size": args.checkpoint_block_size,
            "commit": current_commit(),
            "metadata": best_effort_formal_metadata(),
            "command": public_command(
                "student_scripts/a2k/checkpoint_benchmark.py",
                [
                    "--context-length",
                    str(args.context_length),
                    "--checkpoint-block-size",
                    str(args.checkpoint_block_size or 0),
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
