from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from cs336_basics.Part4.cross_entropy import cross_entropy
from cs336_basics.Part4.gradient_clipping import clip_gradient_norm_
from cs336_basics.Part5.configuration import load_experiment_config
from cs336_basics.Part5.data_loading import sample_batch
from cs336_basics.Part5.experiment_logging import write_json_atomic
from cs336_basics.Part5.training import (
    build_model,
    build_optimizer,
    load_token_dataset,
    resolve_amp_dtype,
    resolve_device,
    resolve_dtype,
    resolve_project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe the largest trainable micro batch size on the current device.")
    parser.add_argument("--config", default="configs/tinystories_baseline.json")
    parser.add_argument("--device", help="Override the configured device.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--upper-bound", type=int, default=4096)
    parser.add_argument("--output", default="logs/batch_size/max_batch_probe.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.start <= 0 or args.upper_bound < args.start:
        raise ValueError("batch-size bounds must satisfy 0 < start <= upper-bound.")

    config = load_experiment_config(resolve_project_path(args.config))
    device = resolve_device(args.device or config.training.device)
    dtype = resolve_dtype(config.training.dtype)
    amp_dtype = resolve_amp_dtype(config.training.amp_dtype)
    if amp_dtype is not None and device.type not in ("cpu", "cuda"):
        raise ValueError("configured bfloat16 autocast requires CPU or CUDA for batch probing.")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(config.model, device=device, dtype=dtype)
    optimizer = build_optimizer(model, config)
    dataset = load_token_dataset(config.data.train_path)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.training.seed)

    def probe(batch_size: int) -> bool:
        inputs: torch.Tensor | None = None
        targets: torch.Tensor | None = None
        loss: torch.Tensor | None = None
        try:
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = sample_batch(
                dataset,
                batch_size,
                config.model.context_length,
                device,
                generator,
            )
            if amp_dtype is None:
                logits = model(inputs)
            else:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            clip_gradient_norm_(model.parameters(), config.optimizer.max_grad_norm)
            optimizer.step()
            _synchronize(device)
            print(f"batch_size={batch_size} status=ok loss={float(loss.detach().float().item()):.6f}", flush=True)
            return True
        except RuntimeError as error:
            if not _is_out_of_memory(error):
                raise
            print(f"batch_size={batch_size} status=out_of_memory", flush=True)
            optimizer.zero_grad(set_to_none=True)
            inputs = None
            targets = None
            loss = None
            error.__traceback__ = None
            gc.collect()
            _empty_cache(device)
            return False

    largest_success = 0
    first_failure = args.upper_bound + 1
    candidate = args.start
    while candidate <= args.upper_bound:
        if probe(candidate):
            largest_success = candidate
            if candidate == args.upper_bound:
                break
            candidate = min(candidate * 2, args.upper_bound)
            if candidate == largest_success:
                break
        else:
            first_failure = candidate
            break

    low = largest_success + 1
    high = min(first_failure - 1, args.upper_bound)
    while low <= high:
        midpoint = (low + high) // 2
        if probe(midpoint):
            largest_success = midpoint
            low = midpoint + 1
        else:
            first_failure = midpoint
            high = midpoint - 1

    search_complete = first_failure <= args.upper_bound
    result = {
        "config": str(Path(args.config)),
        "model": config.to_dict()["model"],
        "device": str(device),
        "dtype": config.training.dtype,
        "amp_dtype": config.training.amp_dtype,
        "context_length": config.model.context_length,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "largest_successful_batch_size": largest_success,
        "first_failing_batch_size": None if first_failure > args.upper_bound else first_failure,
        "upper_bound": args.upper_bound,
        "search_complete": search_complete,
    }
    write_json_atomic(resolve_project_path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if largest_success == 0:
        return 2
    return 0 if search_complete else 3


def _is_out_of_memory(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "memory allocation" in message


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
