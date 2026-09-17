"""A2-K 任务五性能矩阵；实际 CUDA 执行由 Slurm 脚本负责。"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import gc
import time
from pathlib import Path

import torch
import triton.testing

from cs336_systems.a2k.attention import scaled_dot_product_attention
from cs336_systems.a2k.flash_attention import (
    TRITON_NUM_WARPS,
    FlashAttentionTritonFunction,
    triton_num_stages,
)

ALLOCATOR_LIMIT_MIB = 23_552
PREFLIGHT_FREE_MIB = 22 * 1024
HARD_LIMIT_MIB = 24 * 1024
MEMORY_MEASURE_ITERATIONS = 5
TRITON_K_TILE_SIZE = 64
TRITON_Q_TILE_SIZE = 64

FIELDS = ["implementation", "sequence_length", "head_dim", "batch_size", "dtype", "is_causal", "phase",
          "p20_ms", "p50_ms", "p80_ms", "peak_allocated_mib", "peak_reserved_mib", "speedup_vs_eager",
          "status", "error", "q_tile_size", "k_tile_size", "num_warps", "num_stages", "cold_start_ms",
          "triton_num_stages", "timing_note"]


def append(path: Path, values: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: values.get(field, "") for field in FIELDS})


def require_free_memory(minimum_mib: int = PREFLIGHT_FREE_MIB) -> int:
    """正式矩阵开跑前的空闲显存前置检查。

    必须在 allocator_limit() 之前调用：那样量到的才是「还没被本进程占用」的真实
    空闲量。题面要求不足 22 GiB 时等待资源释放，不得缩小 shape——因此这里直接
    抛错，而不是继续用一个缩小的配置跑出无效结果。
    """
    free_bytes, _ = torch.cuda.mem_get_info()
    free_mib = free_bytes / 2**20
    if free_mib < minimum_mib:
        raise RuntimeError(
            f"开跑前可用显存 {free_mib:.0f} MiB < 要求的 {minimum_mib} MiB（22 GiB）；"
            "请等待资源释放，不得缩小 shape"
        )
    return int(free_mib)


def allocator_limit():
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, ALLOCATOR_LIMIT_MIB * 2**20 / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def measure_latency(fn):
    """按题面固定协议测延迟：warmup=100ms、rep=300ms、分位 [0.2, 0.5, 0.8]。"""
    values = triton.testing.do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])
    if torch.is_tensor(values):
        values = values.detach().cpu().flatten().tolist()
    return tuple(float(x) for x in values)


def measure_peak_memory(step, iterations=MEMORY_MEASURE_ITERATIONS):
    """独立测量峰值显存，不依赖 do_bench。

    为什么必须单独测：``do_bench`` 内部有一次按时间计的 warmup（100 ms，可能几十次
    调用），峰值统计会把那部分也算进去，得到的不是单次稳定的峰值。这里改为
    「调用方先 reset 统计 → 跑固定次数 → 读峰值」。

    注意调用方在第一次 ``step()`` 之前就要 reset、且中间不再 reset，这样首次调用
    的分配也会被计入峰值。
    """
    for _ in range(iterations):
        step()
        torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20


#: 任务五固定矩阵：8 个 shape × 3 个 phase × 3 种实现 = 72 个配置。
BATCH_SHAPES = ((512, 64), (512, 128), (2048, 64), (2048, 128),
                (8192, 64), (8192, 128), (16384, 64), (16384, 128))
BATCH_PHASES = ("forward", "backward", "forward_backward")
BATCH_IMPLEMENTATIONS = ("eager", "compiled", "triton")


def run_batch(output: Path) -> None:
    """逐配置跑完整矩阵；每个配置都是一个独立进程。

    独立进程是必须的：allocator 上限要在首次 CUDA allocation 之前设置，而且不同
    shape 的显存缓存、``torch.compile`` 缓存不能相互污染。单个配置失败（例如 OOM）
    不中止整批——失败会由该配置自己写进 CSV 的 status 列。
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    failures: list[str] = []
    for sequence_length, head_dim in BATCH_SHAPES:
        for phase in BATCH_PHASES:
            for implementation in BATCH_IMPLEMENTATIONS:
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--implementation", implementation,
                    "--sequence-length", str(sequence_length),
                    "--head-dim", str(head_dim),
                    "--phase", phase,
                    "--output", str(output),
                ]
                print(" ".join(command), flush=True)
                result = subprocess.run(command, check=False, cwd=Path.cwd())
                if result.returncode != 0:
                    failures.append(f"{sequence_length}/{head_dim}/{phase}/{implementation}")
    print(f"矩阵完成：{len(BATCH_SHAPES) * len(BATCH_PHASES) * len(BATCH_IMPLEMENTATIONS)} 个配置，"
          f"子进程失败 {len(failures)} 个", flush=True)
    for item in failures:
        print(f"  子进程退出码非 0：{item}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("eager", "compiled", "triton"))
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--phase", choices=("forward", "backward", "forward_backward"))
    parser.add_argument("--batch", action="store_true",
                        help="逐配置跑完任务五固定矩阵（每个配置一个独立进程）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch:
        run_batch(args.output)
        return
    missing = [name for name in ("implementation", "sequence_length", "head_dim", "phase")
               if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"单配置模式缺少参数 {missing}；若要跑完整矩阵请用 --batch")

    num_stages = triton_num_stages(args.head_dim)
    row = {"implementation": args.implementation, "sequence_length": args.sequence_length,
           "head_dim": args.head_dim, "batch_size": 1, "dtype": "bfloat16", "is_causal": True,
           "phase": args.phase, "status": "ok",
           "q_tile_size": TRITON_Q_TILE_SIZE if args.implementation == "triton" else "",
           "k_tile_size": TRITON_K_TILE_SIZE if args.implementation == "triton" else "",
           "num_warps": TRITON_NUM_WARPS if args.implementation == "triton" else "",
           "num_stages": num_stages if args.implementation == "triton" else "",
           "triton_num_stages": num_stages if args.implementation == "triton" else ""}
    if not torch.cuda.is_available():
        row.update(status="skip", error="CUDA unavailable")
        append(args.output, row)
        return
    try:
        free_before = require_free_memory()
        fraction = allocator_limit()
        print(f"前置检查通过：开跑前可用显存 {free_before} MiB；"
              f"allocator_fraction={fraction:.6f}，上限 {ALLOCATOR_LIMIT_MIB} MiB，"
              f"24 GiB 硬上限 {HARD_LIMIT_MIB} MiB")
        torch.manual_seed(args.seed)
        device = torch.device("cuda")
        q = torch.randn((1, args.sequence_length, args.head_dim), device=device,
                        dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        grad_output = torch.randn_like(q)

        # N×N causal mask 只被 PyTorch 显式实现用到；Triton attention 在 kernel 内部
        # 用索引比较生成 causal mask，不消费这个二次方张量。因此对 Triton 配置**不创建**
        # 它，否则 Triton 的峰值显存会被这个无用的 mask 主导（N=16384 时 bool 就有 256 MiB），
        # 直接削弱「FlashAttention 显存随序列线性」的结论。
        mask = None
        if args.implementation in ("eager", "compiled"):
            mask = torch.ones((args.sequence_length, args.sequence_length),
                              device=device, dtype=torch.bool).tril()

        compiled = False
        if args.implementation == "eager":
            def fwd():
                return scaled_dot_product_attention(q, k, v, mask)
        elif args.implementation == "compiled":
            module = torch.compile(lambda x, y, z: scaled_dot_product_attention(x, y, z, mask),
                                   backend="inductor", mode="default")

            def fwd():
                return module(q, k, v)
            compiled = True
        else:
            def fwd():
                return FlashAttentionTritonFunction.apply(q, k, v, True)

        def fwd_bwd():
            return torch.autograd.grad(fwd(), (q, k, v), grad_output)

        def backward_on(graph):
            return torch.autograd.grad(graph, (q, k, v), grad_output, retain_graph=True)

        # backward 的测法：把前向图建一次并保留（retain_graph=True），此后只调
        # autograd.grad——重复调用不会重跑前向，因此 p20/p50/p80 是对「纯反向」分布
        # 直接测得的分位数，而不是两个独立分布分位数相减（那样统计上不成立，实测曾
        # 出现 p80 < p50 的乱序）。
        #
        # 同一份保留图也用于显存测量：反向本来就要求前向激活在图里存活，所以任何纯
        # 反向测量都必须带着这份激活，不存在「丢掉它再测」的方案。关键是**不能再额外
        # 建第二张图**——上一版那样做会让 backward 行的峰值反而高于 forward_backward。
        held_graph = None
        cold_start_ms = None
        if args.phase == "forward":
            step = fwd
            row["timing_note"] = "forward"
        elif args.phase == "backward":
            held_graph = fwd()
            step = lambda: backward_on(held_graph)
            row["timing_note"] = "backward (held forward graph, retain_graph=True)"
        else:
            step = fwd_bwd
            row["timing_note"] = "forward_backward"

        # cold-start 与峰值显存的边界必须分清，否则 compiled 行会同时错两处：
        #
        #   1. AOTAutograd 的反向图是在**第一次 autograd.grad()** 时才编译的。只计时
        #      首次 fwd() 会漏掉反向的编译时间（约 2 s），cold_start_ms 不完整；
        #   2. 那一次编译的分配器占用若落在峰值窗口内，会让 compiled backward 行的
        #      peak_reserved 高于 forward_backward 行——实测正是如此。
        #
        # 因此 compiled 配置统一先用一次「不计时、不计峰值」的冷启动步把 forward 与
        # backward 图都编出来，记录合计 wall time；随后 reset 峰值统计，之后的测量才是
        # 稳态延迟与稳态显存。
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if compiled:
            start = time.perf_counter()
            # backward 阶段的前向图已经建好，这一步补编译反向；forward /
            # forward_backward 阶段由首次 step() 触发前向与反向的编译。
            step()
            torch.cuda.synchronize()
            cold_start_ms = (time.perf_counter() - start) * 1000
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        else:
            step()
            torch.cuda.synchronize()

        peak_allocated, peak_reserved = measure_peak_memory(step)

        gc.collect()
        torch.cuda.empty_cache()
        p20, p50, p80 = measure_latency(step)
        torch.cuda.synchronize()
        row.update(p20_ms=f"{p20:.4f}", p50_ms=f"{p50:.4f}", p80_ms=f"{p80:.4f}",
                   peak_allocated_mib=f"{peak_allocated:.3f}",
                   peak_reserved_mib=f"{peak_reserved:.3f}")
        if cold_start_ms is not None:
            row["cold_start_ms"] = f"{cold_start_ms:.4f}"
    except torch.cuda.OutOfMemoryError as exc:
        row.update(status="oom", error=str(exc)[:300])
    except Exception as exc:
        row.update(status="compile_failure" if args.implementation == "compiled" else "failed",
                   error=f"{type(exc).__name__}: {exc}"[:500])
    append(args.output, row)


if __name__ == "__main__":
    main()
