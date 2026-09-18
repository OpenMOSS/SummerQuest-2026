from __future__ import annotations

import argparse
import subprocess
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton.testing

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k.attention import scaled_dot_product_attention as sdpa

ALLOCATOR_LIMIT_MIB = 23_552
WARMUP_MS = 100
REP_MS = 300
QUANTILES = (0.2, 0.5, 0.8)
PHASES = ("forward", "backward", "forward_backward")
CSV_FIELDS = [
    "target", "implementation", "sequence_length", "head_dim", "batch_size", "dtype",
    "is_causal", "phase", "seed", "warmup_ms", "rep_ms", "quantiles",
    "p20_ms", "p50_ms", "p80_ms", "peak_allocated_mib", "peak_reserved_mib",
    "cold_start_ms", "allocator_limit_mib", "allocator_fraction",
    "compile_backend", "compile_mode", "status", "error",
]


class AttentionModule(nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return sdpa(q, k, v, self.mask)


def set_allocator_limit() -> float:
    """在创建 CUDA tensor、模型或优化器前设置 23 GiB allocator 上限。"""
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, ALLOCATOR_LIMIT_MIB * 2**20 / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    """在计时外创建下三角 causal mask。"""
    return torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool).tril()


def quantile_values(fn) -> tuple[float, float, float]:
    """按题面协议使用 Triton do_bench，返回 p20/p50/p80 毫秒。"""
    result = triton.testing.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, quantiles=list(QUANTILES))
    if torch.is_tensor(result):
        values = result.detach().cpu().flatten().tolist()
    elif isinstance(result, (tuple, list)):
        values = [float(value) for value in result]
    else:
        values = [float(result)] * 3
    if not values:
        raise RuntimeError("do_bench 没有返回延迟分位数")
    values = (values + [values[-1]] * 3)[:3]
    return tuple(float(value) for value in values)


def append_row(path: Path, row: dict) -> None:
    """追加固定字段的 CSV 行，避免不同 target 产生不同表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore",
                                lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def append_metadata(path: Path | None, row: dict, command: list[str], fraction: float | None) -> None:
    """写入脱敏的环境和命令 metadata；不写主机名、用户名或 GPU UUID。"""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "command": command,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": row.get("seed"),
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": fraction,
        "warmup_ms": WARMUP_MS,
        "rep_ms": REP_MS,
        "quantiles": list(QUANTILES),
        "row_status": row.get("status"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=True) + "\n")


def base_row(target: str, implementation: str, sequence_length: int, head_dim: int, phase: str, seed: int) -> dict:
    """构造所有 benchmark 行共享的字段。"""
    return {
        "target": target,
        "implementation": implementation,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "batch_size": 1,
        "dtype": "bfloat16" if target == "attention" else "fp32_params+bf16_autocast",
        "is_causal": True,
        "phase": phase,
        "seed": seed,
        "warmup_ms": WARMUP_MS,
        "rep_ms": REP_MS,
        "quantiles": "0.2;0.5;0.8",
        "p20_ms": "", "p50_ms": "", "p80_ms": "",
        "peak_allocated_mib": "", "peak_reserved_mib": "", "cold_start_ms": "",
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB, "allocator_fraction": "",
        "compile_backend": "inductor" if implementation == "compiled" else "",
        "compile_mode": "default" if implementation == "compiled" else "",
        "status": "ok", "error": "",
    }


def run_phase_once(fn, q, k, v, do, phase: str) -> None:
    """执行一次 phase，用于编译冷启动或 eager 预热。"""
    output = fn(q, k, v)
    if phase == "forward":
        return
    torch.autograd.grad(output, (q, k, v), do, retain_graph=(phase == "backward"))


def build_steady_callable(fn, q, k, v, do, phase: str):
    """构造不创建输入的稳定态 callable。"""
    if phase == "forward":
        return lambda: fn(q, k, v)
    if phase == "backward":
        # backward phase 的前向图在计时外创建，计时只覆盖反向。
        output = fn(q, k, v)
        return lambda: torch.autograd.grad(output, (q, k, v), do, retain_graph=True)

    def forward_backward():
        output = fn(q, k, v)
        torch.autograd.grad(output, (q, k, v), do, retain_graph=False)

    return forward_backward


def measure_attention(implementation: str, sequence_length: int, head_dim: int, phase: str, seed: int) -> tuple[dict, float | None]:
    """测量一个显式 attention shape/phase，失败时保留 OOM 或编译错误行。"""
    row = base_row("attention", implementation, sequence_length, head_dim, phase, seed)
    fraction: float | None = None
    try:
        fraction = set_allocator_limit()
        row["allocator_fraction"] = fraction
        torch.manual_seed(seed)
        device = torch.device("cuda")
        requires_grad = phase != "forward"
        q = torch.randn((1, sequence_length, head_dim), device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
        k = torch.randn_like(q, requires_grad=requires_grad)
        v = torch.randn_like(q, requires_grad=requires_grad)
        do = torch.randn_like(q) if requires_grad else None
        mask = make_causal_mask(sequence_length, device)
        eager_module = AttentionModule(mask)
        fn = eager_module
        if implementation == "compiled":
            fn = torch.compile(eager_module, backend="inductor", mode="default")
            start = time.perf_counter()
            run_phase_once(fn, q, k, v, do, phase)
            torch.cuda.synchronize()
            row["cold_start_ms"] = f"{(time.perf_counter() - start) * 1000.0:.4f}"
        else:
            run_phase_once(fn, q, k, v, do, phase)
            torch.cuda.synchronize()

        # 释放冷启动阶段的临时图和 allocator 缓存，再测 steady state。
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        steady = build_steady_callable(fn, q, k, v, do, phase)
        torch.cuda.synchronize()
        p20, p50, p80 = quantile_values(steady)
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        row.update({
            "p20_ms": f"{p20:.4f}", "p50_ms": f"{p50:.4f}", "p80_ms": f"{p80:.4f}",
            "peak_allocated_mib": f"{allocated:.3f}", "peak_reserved_mib": f"{reserved:.3f}",
        })
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
        torch.cuda.empty_cache()
    except Exception as exc:
        row["status"] = "compile_error" if implementation == "compiled" else "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return row, fraction


class SmallTrainingModule(nn.Module):
    """封装 Stanford small 模型，并在 forward 中使用 BF16 autocast。"""
    def __init__(self, compiled: bool):
        super().__init__()
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=512,
            d_model=768,
            num_layers=12,
            num_heads=12,
            d_ff=3072,
            rope_theta=10_000.0,
        ).cuda()
        self.model = torch.compile(model, backend="inductor", mode="default") if compiled else model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model(tokens)


def measure_full_model(implementation: str, phase: str, seed: int) -> tuple[dict, float | None]:
    """测量 small Transformer 的 eager/compiled phase。"""
    row = base_row("small_transformer", implementation, 512, 64, phase, seed)
    row["head_dim"] = 64
    fraction: float | None = None
    model_module = None
    original_attention = None
    try:
        fraction = set_allocator_limit()
        row["allocator_fraction"] = fraction
        torch.manual_seed(seed)
        import cs336_basics.model as model_module

        # 只在当前进程临时替换全局引用，不修改基础源文件。
        original_attention = model_module.scaled_dot_product_attention
        model_module.scaled_dot_product_attention = sdpa
        module = SmallTrainingModule(implementation == "compiled")
        optimizer = AdamW(module.parameters(), lr=1e-3)
        tokens = torch.randint(10_000, (1, 512), device="cuda")
        labels = torch.randint(10_000, (1, 512), device="cuda")

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            if phase == "forward":
                with torch.no_grad():
                    module(tokens)
                return
            loss = cross_entropy(module(tokens), labels)
            loss.backward()
            if phase == "train_step":
                optimizer.step()

        if implementation == "compiled":
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            row["cold_start_ms"] = f"{(time.perf_counter() - start) * 1000.0:.4f}"
        else:
            step()
            torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        p20, p50, p80 = quantile_values(step)
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        row.update({
            "p20_ms": f"{p20:.4f}", "p50_ms": f"{p50:.4f}", "p80_ms": f"{p80:.4f}",
            "peak_allocated_mib": f"{allocated:.3f}", "peak_reserved_mib": f"{reserved:.3f}",
        })
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
        torch.cuda.empty_cache()
    except Exception as exc:
        row["status"] = "compile_error" if implementation == "compiled" else "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if model_module is not None and original_attention is not None:
            model_module.scaled_dot_product_attention = original_attention
    return row, fraction


#: 任务二固定矩阵：与报告 §4 一致，每个配置一个独立 Python 子进程。
BASELINE_SHAPES = ((512, 64), (512, 128), (2048, 64), (2048, 128), (8192, 64), (8192, 128))
COMPILE_SHAPES = ((512, 64), (2048, 128), (8192, 128))
PHASE_CHOICES = ("forward", "backward", "forward_backward")


def run_batch(args) -> None:
    """在单张 RTX 4090 上串行跑完任务二的**一个**矩阵。

    ``--matrix baseline`` 跑 6 个 shape × 3 个 phase 的显式 attention；
    ``--matrix compile`` 跑 3 个代表配置的 eager/compiled 对照以及 small 模型的
    train_step 对照。两个矩阵的产物文件不同，因此分成两次调用，各自写自己的 --output。

    每个配置都用独立的 ``sys.executable`` 子进程执行，进程之间不共享显存缓存与
    ``torch.compile`` 缓存——这是固定矩阵要求的隔离方式，也让每个配置的
    allocator 上限都在首次 CUDA allocation 之前设置。
    """
    if args.matrix is None:
        raise ValueError("批量模式需要 --matrix baseline 或 --matrix compile")
    for output in (args.output, args.metadata_output):
        if output is not None and output.exists():
            output.unlink()

    def spawn(*extra: str) -> None:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--seed", str(args.seed),
            "--warmup-ms", str(args.warmup_ms),
            "--rep-ms", str(args.rep_ms),
            *extra,
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=Path.cwd())

    base = ["--metadata-output", str(args.metadata_output)]
    if args.matrix == "baseline":
        for sequence_length, head_dim in BASELINE_SHAPES:
            for phase in PHASE_CHOICES:
                spawn("--mode", "baseline", "--implementation", "eager",
                      "--sequence-length", str(sequence_length), "--head-dim", str(head_dim),
                      "--phase", phase, "--output", str(args.output), *base)
        return
    for sequence_length, head_dim in COMPILE_SHAPES:
        for implementation in ("eager", "compiled"):
            for phase in PHASE_CHOICES:
                spawn("--mode", "compile", "--implementation", implementation,
                      "--sequence-length", str(sequence_length), "--head-dim", str(head_dim),
                      "--phase", phase, "--output", str(args.output), *base)
    for implementation in ("eager", "compiled"):
        for phase in ("forward", "forward_backward", "train_step"):
            spawn("--mode", "full_model", "--implementation", implementation,
                  "--phase", phase, "--output", str(args.output), *base)


def main() -> None:
    global WARMUP_MS, REP_MS
    parser = argparse.ArgumentParser(description="A2-K 任务二 attention benchmark")
    parser.add_argument("--mode", choices=["baseline", "compile", "full_model"])
    parser.add_argument("--batch", action="store_true",
                        help="跑任务二固定矩阵中的一个矩阵（每个配置一个独立子进程）")
    parser.add_argument("--matrix", choices=("baseline", "compile"),
                        help="配合 --batch 选择要跑的矩阵")
    parser.add_argument("--implementation", choices=["eager", "compiled"], default="eager")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--phase", choices=(*PHASES, "train_step"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--warmup-ms", type=int, default=WARMUP_MS)
    parser.add_argument("--rep-ms", type=int, default=REP_MS)
    args = parser.parse_args()
    WARMUP_MS, REP_MS = args.warmup_ms, args.rep_ms
    if args.batch:
        run_batch(args)
        return
    if args.mode is None or args.phase is None:
        raise ValueError("单配置模式需要 --mode 与 --phase；若要跑完整矩阵请用 --batch")
    if not torch.cuda.is_available():
        raise RuntimeError("任务二正式测量需要 CUDA，请在单张 RTX 4090 上执行")

    if args.mode == "full_model":
        row, fraction = measure_full_model(args.implementation, args.phase, args.seed)
    else:
        if args.sequence_length is None or args.head_dim is None:
            raise ValueError("attention 模式需要 --sequence-length 和 --head-dim")
        row, fraction = measure_attention(args.implementation, args.sequence_length, args.head_dim, args.phase, args.seed)
    append_row(args.output, row)
    append_metadata(args.metadata_output, row, sys.argv, fraction)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
