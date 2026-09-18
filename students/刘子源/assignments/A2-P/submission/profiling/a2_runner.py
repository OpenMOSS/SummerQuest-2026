from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.attention import FlashAttentionPyTorchFunction, FlashAttentionTritonFunction


SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def save(path: str, item: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_metadata() -> dict:
    value = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": "cpu",
        "device_name": None,
        "memory_total_mib": None,
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        value.update(
            {
                "device": "cuda:0",
                "device_name": prop.name,
                "memory_total_mib": round(prop.total_memory / 2**20, 1),
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            }
        )
        try:
            import triton

            value["triton"] = triton.__version__
        except ImportError:
            value["triton"] = None
    return value


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device: torch.device, dtype: str | None):
    if device.type != "cuda" or dtype is None or dtype == "float32":
        return nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, dtype))


def allocator_limit(mib: int | None) -> None:
    if mib is None or not torch.cuda.is_available():
        return
    fraction = mib * 2**20 / torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(fraction, 0)


def memory_summary(device: torch.device) -> dict | None:
    if device.type != "cuda":
        return None
    stats = torch.cuda.memory_stats(device)
    return {
        "active_peak_bytes": stats.get("active_bytes.all.peak", 0),
        "allocated_peak_bytes": stats.get("allocated_bytes.all.peak", 0),
        "reserved_peak_bytes": stats.get("reserved_bytes.all.peak", 0),
        "requested_peak_bytes": stats.get("requested_bytes.all.peak", 0),
        "current_allocated_bytes": torch.cuda.memory_allocated(device),
        "current_reserved_bytes": torch.cuda.memory_reserved(device),
    }


def time_many(step: Callable[[], None], device: torch.device, warmup: int, steps: int) -> dict:
    for _ in range(warmup):
        step()
    cuda_sync(device)
    raw = []
    for _ in range(steps):
        cuda_sync(device)
        t0 = time.perf_counter()
        step()
        cuda_sync(device)
        raw.append(time.perf_counter() - t0)
    mean = statistics.fmean(raw)
    stdev = statistics.stdev(raw) if len(raw) > 1 else 0.0
    ordered = sorted(raw)

    def pct(q: float) -> float:
        where = (len(ordered) - 1) * q
        lo, hi = math.floor(where), math.ceil(where)
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (where - lo)

    return {"raw_seconds": raw, "mean_seconds": mean, "sample_stdev_seconds": stdev, "cv": stdev / mean if mean else None, "p20_seconds": pct(0.2), "p50_seconds": pct(0.5), "p80_seconds": pct(0.8)}


def model(args: argparse.Namespace, device: torch.device) -> BasicsTransformerLM:
    return BasicsTransformerLM(vocab_size=args.vocab_size, context_length=args.context_length, **SIZES[args.model_size]).to(device)


def model_step(args: argparse.Namespace, use_compile: bool = False):
    device = torch.device(args.device)
    m = model(args, device)
    if use_compile:
        m = torch.compile(m, mode="reduce-overhead")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    labels = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)

    def step() -> None:
        if args.mode != "forward":
            opt.zero_grad(set_to_none=True)
        with autocast(device, args.dtype):
            logits = m(tokens)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
        if args.mode in ("forward_backward", "train_step"):
            loss.backward()
        if args.mode == "train_step":
            opt.step()

    return step, device


def run_model(args: argparse.Namespace) -> dict:
    allocator_limit(args.allocator_limit_mib)
    torch.manual_seed(args.seed)
    value = {"kind": "model_benchmark", "model_size": args.model_size, "batch_size": args.batch_size, "context_length": args.context_length, "mode": args.mode, "dtype": args.dtype or "float32", "compile": args.compile, "warmup": args.warmup, "steps": args.steps, "seed": args.seed, "metadata": gpu_metadata()}
    device = torch.device(args.device)
    try:
        step, device = model_step(args, args.compile)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        value.update(time_many(step, device, args.warmup, args.steps))
        value["memory"] = memory_summary(device)
        value["status"] = "ok"
    except RuntimeError as err:
        value.update({"status": "oom" if "out of memory" in str(err).lower() else "error", "error": str(err), "memory": memory_summary(device)})
    save(args.output, value)
    return value


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        score = score.masked_fill(torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1), float("-inf"))
    return torch.softmax(score, dim=-1) @ v


def attention_step(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    q = torch.randn(args.batch_size, args.sequence_length, args.dimension, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    if args.implementation == "compiled":
        fn = torch.compile(attention_reference, mode="reduce-overhead")
    elif args.implementation == "triton":
        fn = lambda a, b, c, d: FlashAttentionTritonFunction.apply(a, b, c, d)
    elif args.implementation == "tiled_pytorch":
        fn = lambda a, b, c, d: FlashAttentionPyTorchFunction.apply(a, b, c, d)
    else:
        fn = attention_reference

    def step() -> None:
        q.grad = k.grad = v.grad = None
        out = fn(q, k, v, args.causal)
        if args.phase != "forward":
            out.float().square().mean().backward()

    return step, device


def run_attention(args: argparse.Namespace) -> dict:
    allocator_limit(args.allocator_limit_mib)
    torch.manual_seed(args.seed)
    value = {"kind": "attention", "implementation": args.implementation, "batch_size": args.batch_size, "sequence_length": args.sequence_length, "dimension": args.dimension, "phase": args.phase, "causal": args.causal, "dtype": args.dtype, "warmup": args.warmup, "steps": args.steps, "seed": args.seed, "metadata": gpu_metadata()}
    device = torch.device(args.device)
    try:
        step, device = attention_step(args)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if args.phase == "backward":
            # A new graph is created outside each timed region, so the timing is backward-only.
            device = torch.device(args.device)
            dtype = getattr(torch, args.dtype)
            q = torch.randn(args.batch_size, args.sequence_length, args.dimension, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)
            if args.implementation == "compiled":
                fn = torch.compile(attention_reference, mode="reduce-overhead")
            elif args.implementation == "triton":
                fn = lambda a, b, c, d: FlashAttentionTritonFunction.apply(a, b, c, d)
            elif args.implementation == "tiled_pytorch":
                fn = lambda a, b, c, d: FlashAttentionPyTorchFunction.apply(a, b, c, d)
            else:
                fn = attention_reference

            def make_loss() -> torch.Tensor:
                q.grad = k.grad = v.grad = None
                return fn(q, k, v, args.causal).float().square().mean()

            for _ in range(args.warmup):
                make_loss().backward()
            cuda_sync(device)
            raw = []
            for _ in range(args.steps):
                loss = make_loss()
                cuda_sync(device)
                t0 = time.perf_counter()
                loss.backward()
                cuda_sync(device)
                raw.append(time.perf_counter() - t0)
            mean = statistics.fmean(raw)
            stdev = statistics.stdev(raw) if len(raw) > 1 else 0.0
            ordered = sorted(raw)
            value.update({"raw_seconds": raw, "mean_seconds": mean, "sample_stdev_seconds": stdev, "cv": stdev / mean if mean else None, "p20_seconds": ordered[max(0, round((len(ordered) - 1) * .2))], "p50_seconds": statistics.median(raw), "p80_seconds": ordered[min(len(ordered) - 1, round((len(ordered) - 1) * .8))]})
        else:
            value.update(time_many(step, device, args.warmup, args.steps))
        value["memory"] = memory_summary(device)
        value["status"] = "ok"
    except RuntimeError as err:
        value.update({"status": "oom" if "out of memory" in str(err).lower() else "error", "error": str(err), "memory": memory_summary(device)})
    save(args.output, value)
    return value


def checkpoint_step(args: argparse.Namespace):
    from torch.utils.checkpoint import checkpoint_sequential

    device = torch.device(args.device)
    m = model(args, device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    labels = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with autocast(device, args.dtype):
            hidden = m.token_embeddings(tokens)
            if args.block_size:
                segments = math.ceil(len(m.layers) / args.block_size)
                hidden = checkpoint_sequential(m.layers, segments, hidden, use_reentrant=False)
            else:
                for layer in m.layers:
                    hidden = layer(hidden)
            logits = m.lm_head(m.ln_final(hidden))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
        loss.backward()
        opt.step()

    return step, device


def run_checkpoint(args: argparse.Namespace) -> dict:
    allocator_limit(args.allocator_limit_mib)
    torch.manual_seed(args.seed)
    value = {"kind": "checkpoint", "model_size": args.model_size, "batch_size": args.batch_size, "context_length": args.context_length, "block_size": args.block_size, "dtype": args.dtype, "warmup": args.warmup, "steps": args.steps, "seed": args.seed, "metadata": gpu_metadata()}
    device = torch.device(args.device)
    try:
        step, device = checkpoint_step(args)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        value.update(time_many(step, device, args.warmup, args.steps))
        value["memory"] = memory_summary(device)
        value["status"] = "ok"
    except RuntimeError as err:
        value.update({"status": "oom" if "out of memory" in str(err).lower() else "error", "error": str(err), "memory": memory_summary(device)})
    save(args.output, value)
    return value


def run_mixed(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    accum = []
    for acc_dtype, term_dtype, cast_before_add in ((torch.float32, torch.float32, False), (torch.float16, torch.float16, False), (torch.float32, torch.float16, False), (torch.float32, torch.float16, True)):
        x = torch.tensor(0, dtype=acc_dtype, device=device)
        for _ in range(1000):
            y = torch.tensor(0.01, dtype=term_dtype, device=device)
            x += y.float() if cast_before_add else y
        accum.append({"accumulator": str(acc_dtype), "term": str(term_dtype), "cast_before_add": cast_before_add, "value": float(x.cpu())})

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(32, 10, bias=False)
            self.ln = torch.nn.LayerNorm(10)
            self.fc2 = torch.nn.Linear(10, 5, bias=False)
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor):
            one = self.relu(self.fc1(x))
            two = self.ln(one)
            return one, two, self.fc2(two)

    toy = ToyModel().to(device)
    x = torch.randn(4, 32, device=device)
    labels = torch.randint(0, 5, (4,), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        fc, norm, logits = toy(x)
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    value = {"kind": "mixed_precision", "accumulation": accum, "toy_dtypes": {"parameters": str(next(toy.parameters()).dtype), "fc1_output": str(fc.dtype), "layer_norm_output": str(norm.dtype), "logits": str(logits.dtype), "loss": str(loss.dtype), "gradient": str(next(toy.parameters()).grad.dtype)}, "metadata": gpu_metadata(), "status": "ok"}
    save(args.output, value)
    return value


def run_memory(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    value = {"kind": "memory_profile", "model_size": args.model_size, "batch_size": args.batch_size, "context_length": args.context_length, "mode": args.mode, "dtype": args.dtype, "metadata": gpu_metadata(), "stages": []}
    try:
        m = model(args, device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
        labels = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)

        def record(name: str) -> None:
            cuda_sync(device)
            stats = torch.cuda.memory_stats(device)
            value["stages"].append({"stage": name, "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device), "active_bytes": stats.get("active_bytes.all.current", 0)})

        with autocast(device, args.dtype):
            warm_logits = m(tokens)
            warm_loss = F.cross_entropy(warm_logits.reshape(-1, warm_logits.shape[-1]).float(), labels.reshape(-1))
        if args.mode == "train_step":
            warm_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        cuda_sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        record("after_warmup")
        with autocast(device, args.dtype):
            logits = m(tokens)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
        record("after_forward")
        if args.mode == "train_step":
            loss.backward()
            record("after_backward")
            opt.step()
            record("after_optimizer")
        snapshot_largest = None
        try:
            snapshot = torch.cuda.memory._snapshot()
            sizes = [block["size"] for segment in snapshot["segments"] for block in segment["blocks"] if block["state"] == "active_allocated"]
            snapshot_largest = max(sizes, default=0)
        except Exception as exc:
            value["snapshot_summary_error"] = repr(exc)
        value.update({"status": "ok", "memory": memory_summary(device), "largest_active_allocation_bytes": snapshot_largest})
    except RuntimeError as err:
        value.update({"status": "oom" if "out of memory" in str(err).lower() else "error", "error": str(err), "memory": memory_summary(device)})
    save(args.output, value)
    return value


def run_correctness(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    rows = []
    impls = {"tiled_pytorch": FlashAttentionPyTorchFunction, "triton": FlashAttentionTritonFunction}
    for seed in args.seeds:
        for dim in args.dimensions:
            for causal in (False, True):
                torch.manual_seed(seed)
                q = torch.randn(1, args.sequence_length, dim, device=device, dtype=torch.float32, requires_grad=True)
                k = torch.randn_like(q, requires_grad=True)
                v = torch.randn_like(q, requires_grad=True)
                grad = torch.randn_like(q)
                ref = attention_reference(q, k, v, causal)
                ref_lse = torch.logsumexp((q @ k.transpose(-2, -1)) / math.sqrt(dim), dim=-1)
                if causal:
                    n = args.sequence_length
                    s = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
                    ref_lse = torch.logsumexp(s.masked_fill(torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), 1), float("-inf")), dim=-1)
                ref.backward(grad, retain_graph=True)
                ref_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
                for name, fn in impls.items():
                    q.grad = k.grad = v.grad = None
                    out = fn.apply(q, k, v, causal)
                    saved = [item for item in out.grad_fn.saved_tensors if item.shape == (1, args.sequence_length)]
                    out.backward(grad)
                    diffs = [float((out - ref).abs().max()), float((saved[0] - ref_lse).abs().max())]
                    diffs.extend(float((got - want).abs().max()) for got, want in zip((q.grad, k.grad, v.grad), ref_grads))
                    rows.append({"seed": seed, "dimension": dim, "causal": causal, "implementation": name, "max_abs_o": diffs[0], "max_abs_lse": diffs[1], "max_abs_dq": diffs[2], "max_abs_dk": diffs[3], "max_abs_dv": diffs[4], "status": "pass" if max(diffs) <= args.atol else "fail"})
    value = {"kind": "flash_correctness", "sequence_length": args.sequence_length, "atol": args.atol, "rows": rows, "metadata": gpu_metadata(), "status": "ok" if all(row["status"] == "pass" for row in rows) else "fail"}
    save(args.output, value)
    return value


def run_profile(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    value = {"kind": "profile", "model_size": args.model_size, "batch_size": args.batch_size, "context_length": args.context_length, "metadata": gpu_metadata()}
    try:
        import cs336_basics.model as basics_model

        original_attention = basics_model.scaled_dot_product_attention

        def marked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            with torch.profiler.record_function("attention/scores"):
                scores = Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])
                if mask is not None:
                    scores = scores.masked_fill(~mask, float("-inf"))
            with torch.profiler.record_function("attention/softmax"):
                probs = torch.softmax(scores, dim=-1)
            with torch.profiler.record_function("attention/value"):
                return probs @ V

        basics_model.scaled_dot_product_attention = marked_attention
        m = model(args, device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
        labels = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)

        def full_step(profiled: bool) -> None:
            opt.zero_grad(set_to_none=True)
            if profiled:
                with torch.profiler.record_function("forward"):
                    logits = m(tokens)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
                with torch.profiler.record_function("backward"):
                    loss.backward()
                with torch.profiler.record_function("optimizer"):
                    opt.step()
            else:
                logits = m(tokens)
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1)).backward()
                opt.step()

        for _ in range(args.warmup):
            full_step(False)
        cuda_sync(device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            with torch.profiler.record_function("profile/measure"):
                full_step(True)
        cuda_sync(device)
        basics_model.scaled_dot_product_attention = original_attention
        rows = []
        for item in prof.key_averages():
            cuda_total = getattr(item, "cuda_time_total", getattr(item, "device_time_total", 0.0))
            self_cuda = getattr(item, "self_cuda_time_total", getattr(item, "self_device_time_total", 0.0))
            row = {"name": item.key, "calls": item.count, "cpu_total_us": item.cpu_time_total, "cuda_total_us": cuda_total, "self_cuda_us": self_cuda}
            rows.append(row)
        rows.sort(key=lambda x: x["cuda_total_us"], reverse=True)
        value.update({"status": "ok", "top_ops": rows[:100], "memory": memory_summary(device)})
    except RuntimeError as err:
        value.update({"status": "oom" if "out of memory" in str(err).lower() else "error", "error": str(err), "memory": memory_summary(device)})
    save(args.output, value)
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--allocator-limit-mib", type=int)
    timing = argparse.ArgumentParser(add_help=False)
    timing.add_argument("--warmup", type=int, default=5)
    timing.add_argument("--steps", type=int, default=10)
    shape = argparse.ArgumentParser(add_help=False)
    shape.add_argument("--model-size", choices=SIZES, default="small")
    shape.add_argument("--vocab-size", type=int, default=10_000)
    shape.add_argument("--context-length", type=int, default=512)
    shape.add_argument("--batch-size", type=int, default=4)
    shape.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    m = sub.add_parser("model", parents=[common, timing, shape])
    m.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="train_step")
    m.add_argument("--compile", action="store_true")
    c = sub.add_parser("checkpoint", parents=[common, timing, shape])
    c.add_argument("--block-size", type=int, default=0)
    a = sub.add_parser("attention", parents=[common, timing])
    a.add_argument("--implementation", choices=["eager", "compiled", "tiled_pytorch", "triton"], default="eager")
    a.add_argument("--phase", choices=["forward", "backward", "forward_backward"], default="forward_backward")
    a.add_argument("--batch-size", type=int, default=1)
    a.add_argument("--sequence-length", type=int, default=512)
    a.add_argument("--dimension", type=int, default=64)
    a.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    a.add_argument("--causal", action="store_true")
    sub.add_parser("mixed", parents=[common])
    mem = sub.add_parser("memory", parents=[common, shape])
    mem.add_argument("--mode", choices=["forward", "train_step"], default="train_step")
    ok = sub.add_parser("correctness", parents=[common])
    ok.add_argument("--sequence-length", type=int, default=128)
    ok.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ok.add_argument("--dimensions", nargs="+", type=int, default=[32, 64, 128])
    ok.add_argument("--atol", type=float, default=1e-2)
    prof = sub.add_parser("profile", parents=[common, shape])
    prof.add_argument("--warmup", type=int, default=5)
    prof.add_argument("--steps", type=int, default=1)
    prof.set_defaults(mode="train_step", compile=False)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "model":
        run_model(args)
    elif args.command == "checkpoint":
        run_checkpoint(args)
    elif args.command == "attention":
        run_attention(args)
    elif args.command == "mixed":
        run_mixed(args)
    elif args.command == "memory":
        run_memory(args)
    elif args.command == "correctness":
        run_correctness(args)
    else:
        run_profile(args)


if __name__ == "__main__":
    main()
