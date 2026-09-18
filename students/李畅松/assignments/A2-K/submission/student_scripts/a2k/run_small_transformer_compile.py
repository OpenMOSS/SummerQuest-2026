"""Compare eager and torch.compile on the Stanford small Transformer."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

from profiling.benchmark import MODEL_CONFIGS


def event_time(fn, steps: int) -> list[float]:
    values = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    config = MODEL_CONFIGS["small"]
    base = BasicsTransformerLM(args.vocab_size, args.context_length, **config).to(device)
    compiled_model = copy.deepcopy(base)
    compiled_model.forward = torch.compile(compiled_model.forward, fullgraph=True)
    inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16}[args.dtype]

    def make_step(model, mode):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step():
            if mode != "forward":
                optimizer.zero_grad(set_to_none=True)
            context = torch.no_grad() if mode == "forward" else torch.enable_grad()
            with context, torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
            if mode != "forward":
                loss.backward()
            if mode == "train_step":
                optimizer.step()
        return step

    result = {"status": "ok", "config": vars(args), "environment": {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }, "models": {}}
    for name, model in (("eager", base), ("compiled", compiled_model)):
        result["models"][name] = {}
        for mode in ("forward", "forward_backward", "train_step"):
            step = make_step(model, mode)
            torch.cuda.synchronize()
            cold_start_ms = None
            if name == "compiled":
                start = time.perf_counter()
                step()
                torch.cuda.synchronize()
                cold_start_ms = (time.perf_counter() - start) * 1000
            for _ in range(args.warmup):
                step()
            timings = event_time(step, args.steps)
            result["models"][name][mode] = {
                "cold_start_ms": cold_start_ms,
                "timings_ms": timings,
                "mean_ms": statistics.mean(timings),
                "p50_ms": statistics.median(timings),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="local_results/a2k/small_transformer_compile.json")
    main(parser.parse_args())
