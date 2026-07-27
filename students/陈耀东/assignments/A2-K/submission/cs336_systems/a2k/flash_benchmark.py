"""扫描 FlashAttention-2 的正式性能矩阵。

默认性能矩阵固定为 batch=1、BF16、causal=True，扫描
sequence=512/2048/8192、head dimension=64/128，并比较 eager PyTorch、
compiled PyTorch 和 student Triton FlashAttention。长序列边界可通过命令行
额外加入 16384。每条结果包含 forward、backward、forward+backward 三种
延迟、p20/p50/p80、阶段峰值显存和 OOM 状态。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from cs336_systems.a2k.experiment_utils import configure_cuda_allocator, hardware_metadata, summarize_samples
from cs336_systems.a2k.flash_attention import TritonFlashAttention, _select_triton_config, standard_attention


try:
    import triton
except ImportError:
    triton = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_LENGTHS = (512, 2048, 8192)
HEAD_DIMENSIONS = (64, 128)
IMPLEMENTATIONS = ("pytorch", "compiled", "triton")
DTYPES = ("bf16",)


def resolve_attention(name: str):
    """把实现名称解析成统一的四参数函数。"""

    if name == "pytorch":
        return standard_attention
    if name == "compiled":
        return torch.compile(standard_attention, fullgraph=True)
    return TritonFlashAttention.apply


def synchronize() -> None:
    """在每一个样本边界同步 CUDA，避免把异步提交时间当作执行时间。"""

    torch.cuda.synchronize()


def timed_samples(function, *, rep_ms: int, minimum_repeats: int) -> list[float]:
    """用 CUDA event 按时间预算收集逐次延迟。"""

    samples: list[float] = []
    window_start = time.perf_counter()
    while len(samples) < minimum_repeats or (time.perf_counter() - window_start) * 1000 < rep_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del result
    return samples


def warmup(function, *, warmup_ms: int, minimum_repeats: int) -> int:
    """运行 warm-up 时间窗口，并返回实际执行次数。"""

    count = 0
    window_start = time.perf_counter()
    while count < minimum_repeats or (time.perf_counter() - window_start) * 1000 < warmup_ms:
        result = function()
        synchronize()
        del result
        count += 1
    return count


def no_grad_call(function) -> None:
    """执行纯 forward，不创建 autograd 图。"""

    with torch.no_grad():
        result = function()
    del result


def backward_once(output: torch.Tensor, grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """在固定图上执行一次 backward。"""

    q.grad = k.grad = v.grad = None
    output.backward(grad_output, retain_graph=True)


def forward_backward_once(attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_output: torch.Tensor, causal: bool) -> None:
    """重新创建图并完成一次完整 forward + backward。"""

    q.grad = k.grad = v.grad = None
    output = attention(q, k, v, causal)
    output.backward(grad_output)
    del output


def reset_phase_peak() -> None:
    """清除前一阶段的峰值统计。"""

    torch.cuda.reset_peak_memory_stats()


def phase_memory() -> dict[str, float]:
    """返回当前阶段显存峰值，单位 MiB。"""

    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    """测量一条 FlashAttention shape 的三个执行边界。"""

    if not torch.cuda.is_available():
        raise RuntimeError("flash benchmark requires CUDA")
    # 这一行必须早于 Q/K/V 的 CUDA allocation；正式 A2-K 使用 23 GiB。
    allocator = configure_cuda_allocator(torch.device("cuda"), args.allocator_limit_gib)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    shape = (1, args.sequence_length, args.head_dimension)
    torch.manual_seed(args.seed)
    q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad_output = torch.randn_like(q)
    attention = resolve_attention(args.implementation)
    compile_cold_start_ms = None
    if args.implementation == "compiled":
        synchronize()
        start = time.perf_counter_ns()
        no_grad_call(lambda: attention(q, k, v, args.causal))
        synchronize()
        compile_cold_start_ms = (time.perf_counter_ns() - start) / 1_000_000

    phases: dict[str, dict[str, Any]] = {}
    try:
        def forward_call() -> None:
            no_grad_call(lambda: attention(q, k, v, args.causal))

        warmup_count = warmup(
            forward_call,
            warmup_ms=args.warmup_ms,
            minimum_repeats=args.warmup,
        )
        reset_phase_peak()
        forward_samples = timed_samples(
            forward_call,
            rep_ms=args.rep_ms,
            minimum_repeats=args.repeats,
        )
        phases["forward"] = {
            "status": "passed",
            "warmup_count": warmup_count,
            **summarize_samples(forward_samples),
            "memory": phase_memory(),
        }
    except torch.cuda.OutOfMemoryError as error:
        phases["forward"] = {"status": "oom", "error": str(error)}
        torch.cuda.empty_cache()

    memory_before_backward_mib = None
    output_finite = None
    if phases["forward"]["status"] == "passed":
        try:
            output = attention(q, k, v, args.causal)
            synchronize()
            memory_before_backward_mib = torch.cuda.memory_allocated() / 1024**2
            output_finite = bool(torch.isfinite(output).all().item())
            def backward_call() -> None:
                backward_once(output, grad_output, q, k, v)

            warmup_count = warmup(
                backward_call,
                warmup_ms=args.warmup_ms,
                minimum_repeats=args.warmup,
            )
            reset_phase_peak()
            backward_samples = timed_samples(
                backward_call,
                rep_ms=args.rep_ms,
                minimum_repeats=args.repeats,
            )
            phases["backward"] = {
                "status": "passed",
                "warmup_count": warmup_count,
                **summarize_samples(backward_samples),
                "memory": phase_memory(),
            }
            del output
        except torch.cuda.OutOfMemoryError as error:
            phases["backward"] = {"status": "oom", "error": str(error)}
            if "output" in locals():
                del output
            q.grad = k.grad = v.grad = None
            torch.cuda.empty_cache()
    else:
        phases["backward"] = {"status": "skipped_after_forward_oom"}

    try:
        def end_to_end() -> None:
            forward_backward_once(attention, q, k, v, grad_output, args.causal)

        warmup_count = warmup(
            end_to_end,
            warmup_ms=args.warmup_ms,
            minimum_repeats=args.warmup,
        )
        reset_phase_peak()
        end_to_end_samples = timed_samples(
            end_to_end,
            rep_ms=args.rep_ms,
            minimum_repeats=args.repeats,
        )
        phases["forward_backward"] = {
            "status": "passed",
            "warmup_count": warmup_count,
            **summarize_samples(end_to_end_samples),
            "memory": phase_memory(),
        }
        # 旧脚本和旧表格使用 end_to_end；保留别名但不重复测量。
        phases["end_to_end"] = phases["forward_backward"]
    except torch.cuda.OutOfMemoryError as error:
        phases["forward_backward"] = {"status": "oom", "error": str(error)}
        phases["end_to_end"] = phases["forward_backward"]
        q.grad = k.grad = v.grad = None
        torch.cuda.empty_cache()
    return {
        "event": "flashattention_benchmark",
        "status": "passed" if all(phase["status"] == "passed" for phase in phases.values()) else "partial_oom",
        "implementation": args.implementation,
        "dtype": args.dtype,
        "batch_size": 1,
        "causal": args.causal,
        "sequence_length": args.sequence_length,
        "head_dimension": args.head_dimension,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "compile_cold_start_ms": compile_cold_start_ms,
        "forward": phases["forward"],
        "backward": phases["backward"],
        "forward_backward": phases["forward_backward"],
        "end_to_end": phases["end_to_end"],
        "memory_before_backward_mib": memory_before_backward_mib,
        "output_finite": output_finite,
        "environment": {
            **hardware_metadata("cuda"),
            "triton": triton.__version__ if triton is not None else None,
            "triton_config": (
                {"query_tile": _select_triton_config(args.head_dimension)[0],
                 "key_tile": _select_triton_config(args.head_dimension)[1],
                 "num_warps": _select_triton_config(args.head_dimension)[2],
                 "num_stages": 2}
                if args.implementation == "triton" else None
            ),
        },
        "allocator": allocator,
    }


def oom_result(args: argparse.Namespace, error: torch.cuda.OutOfMemoryError) -> dict[str, Any]:
    """保存普通 Attention 或 compiled backward 的 OOM 边界。"""

    return {
        "event": "flashattention_benchmark",
        "status": "oom",
        "implementation": args.implementation,
        "dtype": args.dtype,
        "batch_size": 1,
        "causal": args.causal,
        "sequence_length": args.sequence_length,
        "head_dimension": args.head_dimension,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "allocator_limit_gib": args.allocator_limit_gib,
        "environment": {
            "triton": triton.__version__ if triton is not None else None,
            "triton_config": (
                {
                    "query_tile": _select_triton_config(args.head_dimension)[0],
                    "key_tile": _select_triton_config(args.head_dimension)[1],
                    "num_warps": _select_triton_config(args.head_dimension)[2],
                    "num_stages": 2,
                }
                if args.implementation == "triton" else None
            ),
        },
        "error": str(error),
    }


@dataclass(frozen=True)
class FlashCase:
    implementation: str
    dtype: str
    sequence_length: int
    head_dimension: int
    causal: bool

    @property
    def case_id(self) -> str:
        causal_label = "causal" if self.causal else "noncausal"
        return f"{self.implementation}-{self.dtype}-s{self.sequence_length}-d{self.head_dimension}-{causal_label}"


def run_matrix(args: argparse.Namespace) -> int:
    """用独立子进程执行完整矩阵，支持 ``--resume`` 续跑。"""

    run_dir = args.run_dir or REPO_ROOT / "artifacts" / "runs" / datetime.now().astimezone().strftime(
        "%Y%m%d-%H%M%S-flash-matrix"
    )
    # 外层 formal runner 会先创建 run_dir；目录生命周期不能由 --resume 控制。
    # 是否跳过已经完成的 case 仍由 results.jsonl 和 --resume 决定。
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    completed_ids: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_ids.add(json.loads(line)["case_id"])
    cases = [
        FlashCase(implementation, dtype, sequence_length, head_dimension, args.causal)
        for implementation in args.implementations
        for dtype in args.dtypes
        for sequence_length in args.sequence_lengths
        for head_dimension in args.head_dimensions
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"cases": [{"case_id": case.case_id, **asdict(case)} for case in cases]}, indent=2),
        encoding="utf-8",
    )

    process_errors = 0
    for index, case in enumerate(cases, start=1):
        if case.case_id in completed_ids:
            print(f"[{index}/{len(cases)}] skip {case.case_id}", flush=True)
            continue
        case_dir = run_dir / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "result.jsonl"
        command = [
            sys.executable,
            "-m",
            "cs336_systems.a2k.flash_benchmark",
            "single",
            "--implementation",
            case.implementation,
            "--dtype",
            case.dtype,
            "--sequence-length",
            str(case.sequence_length),
            "--head-dimension",
            str(case.head_dimension),
            "--causal" if args.causal else "--non-causal",
            "--warmup-ms",
            str(args.warmup_ms),
            "--rep-ms",
            str(args.rep_ms),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--allow-oom",
            "--output",
            str(result_path),
        ]
        if args.allocator_limit_gib is not None:
            command.extend(("--allocator-limit-gib", str(args.allocator_limit_gib)))
        (case_dir / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if result_path.exists() and result_path.stat().st_size:
            result = json.loads(result_path.read_text(encoding="utf-8").splitlines()[-1])
        else:
            result = {"event": "flashattention_benchmark", "status": "process_error", "stderr_tail": completed.stderr[-4000:]}
            process_errors += 1
        result.update(case_id=case.case_id, return_code=completed.returncode)
        with results_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(cases)}] {case.case_id}: {result['status']}", flush=True)

    (run_dir / "status.json").write_text(
        json.dumps({"process_errors": process_errors, "case_count": len(cases)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"run_dir={run_dir}")
    return int(process_errors > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention-2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single")
    single.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    single.add_argument("--dtype", choices=DTYPES, required=True)
    single.add_argument("--sequence-length", type=int, required=True)
    single.add_argument("--head-dimension", type=int, required=True)
    single.add_argument("--causal", dest="causal", action="store_true", default=True)
    single.add_argument("--non-causal", dest="causal", action="store_false")
    single.add_argument("--warmup-ms", type=int, default=100)
    single.add_argument("--rep-ms", type=int, default=300)
    single.add_argument("--warmup", type=int, default=3)
    single.add_argument("--repeats", type=int, default=5)
    single.add_argument("--seed", type=int, default=2026)
    single.add_argument("--allocator-limit-gib", type=float, default=None)
    single.add_argument("--allow-oom", action="store_true")
    single.add_argument("--output", type=Path)

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS))
    matrix.add_argument("--dtypes", nargs="+", choices=("fp32", "bf16"), default=list(DTYPES))
    matrix.add_argument("--sequence-lengths", nargs="+", type=int, default=list(SEQUENCE_LENGTHS))
    matrix.add_argument("--head-dimensions", nargs="+", type=int, default=list(HEAD_DIMENSIONS))
    matrix.add_argument("--causal", dest="causal", action="store_true", default=True)
    matrix.add_argument("--non-causal", dest="causal", action="store_false")
    matrix.add_argument("--warmup-ms", type=int, default=100)
    matrix.add_argument("--rep-ms", type=int, default=300)
    matrix.add_argument("--warmup", type=int, default=3)
    matrix.add_argument("--repeats", type=int, default=5)
    matrix.add_argument("--allocator-limit-gib", type=float, default=None)
    matrix.add_argument("--run-dir", type=Path)
    matrix.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "matrix":
        return run_matrix(args)
    try:
        result = run_single(args)
        exit_code = 0
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        result = oom_result(args, error)
        exit_code = 0 if args.allow_oom else 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
