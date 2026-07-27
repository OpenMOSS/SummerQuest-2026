"""标准 PyTorch Attention 与 ``torch.compile`` 的 A2-K 实验矩阵。

正式矩阵固定为 ``batch=1``、``BF16``、``causal=True``，扫描
``sequence length = 512/2048/8192`` 与 ``head dimension = 64/128``。
每个 case 分别测量纯 forward、固定计算图的 backward，以及一次完整
``forward + backward``。计时使用 CUDA event，warm-up 和 measurement 采用
100ms/300ms 的时间预算；OOM、阶段显存和编译冷启动都会落盘。

矩阵模式会为每个 shape/实现启动独立 Python 子进程。这样某个二次显存配置
OOM 后，进程退出即可彻底释放 CUDA allocator 和编译缓存，不会污染下一行。
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
from cs336_systems.a2k.flash_attention import standard_attention


REPO_ROOT = Path(__file__).resolve().parents[2]
MIB = 1024**2
HEAD_DIMENSIONS = (64, 128)
SEQUENCE_LENGTHS = (512, 2048, 8192)
IMPLEMENTATIONS = ("eager", "compiled")


def synchronize() -> None:
    """在每次计时边界等待 GPU，防止只测到异步 kernel 提交。"""

    torch.cuda.synchronize()


def summarize(samples: list[float]) -> dict[str, float]:
    """保留原始样本之外，再计算可直接制表的统计量。"""
    summary = summarize_samples(samples)
    summary["median_ms"] = summary["p50_ms"]
    return summary


def timed_samples(function, *, rep_ms: int, minimum_repeats: int) -> list[float]:
    """用 CUDA event 收集一个 measurement 时间窗口内的逐次延迟。

    ``torch.cuda.Event.elapsed_time`` 只计算 GPU stream 上两次 event 之间的
    时间，不会把 Python 调度和 CPU 睡眠误当成 kernel 执行时间。仍然在每个
    样本结束后同步，是为了让异常和显存状态都在当前样本内完成。
    """

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


def warmup_for_budget(function, *, warmup_ms: int, minimum_repeats: int) -> int:
    """运行 warm-up 时间窗口，并返回实际执行次数。"""

    count = 0
    window_start = time.perf_counter()
    while count < minimum_repeats or (time.perf_counter() - window_start) * 1000 < warmup_ms:
        result = function()
        synchronize()
        del result
        count += 1
    return count


def phase_memory() -> dict[str, float]:
    """读取当前阶段的 active/reserved 与峰值显存。"""

    return cuda_memory()


def reset_phase_peak() -> None:
    """让下一个阶段的峰值不被前一个阶段的临时张量污染。"""

    torch.cuda.reset_peak_memory_stats()


def _no_grad_call(function) -> None:
    """执行纯 forward；它不构建反向图，适合单独的 forward 阶段。"""

    with torch.no_grad():
        result = function()
    del result


def _backward_once(output: torch.Tensor, grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """在固定 forward 图上测量一次 backward，并清除叶子梯度。"""

    q.grad = k.grad = v.grad = None
    output.backward(grad_output, retain_graph=True)


def _forward_backward_once(
    attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
) -> None:
    """建立一张新图并完成一次 forward + backward。"""

    q.grad = k.grad = v.grad = None
    output = attention(q, k, v, causal)
    output.backward(grad_output)
    del output


def cuda_memory() -> dict[str, float]:
    """记录当前 active/reserved 与峰值显存，单位统一为 MiB。"""

    return {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
    }


def build_attention(implementation: str):
    """返回 eager 或编译后的同一数学实现，保证比较只改变编译开关。"""

    if implementation == "eager":
        return standard_attention
    return torch.compile(standard_attention, fullgraph=True)


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    """执行一条 shape/实现配置的三个阶段测量。"""

    if not torch.cuda.is_available():
        raise RuntimeError("attention benchmark requires CUDA")
    # 正式 A2-K 进程必须在创建 Q/K/V 前设置 allocator 上限。
    allocator = configure_cuda_allocator(torch.device("cuda"), args.allocator_limit_gib)
    torch.manual_seed(args.seed)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    attention = build_attention(args.implementation)
    shape = (args.batch_size, args.sequence_length, args.head_dimension)
    q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad_output = torch.randn_like(q)

    # compiled 的第一次调用包含图捕获、代码生成和编译缓存建立。它不能混入
    # steady-state 延迟，因此单独记录 cold start；eager 则没有这个阶段。
    compile_cold_start_ms = None
    if args.implementation == "compiled":
        synchronize()
        start = time.perf_counter_ns()
        with torch.no_grad():
            cold_output = attention(q, k, v, args.causal)
        synchronize()
        compile_cold_start_ms = (time.perf_counter_ns() - start) / 1_000_000
        del cold_output

    phase_results: dict[str, Any] = {}
    try:
        def forward_call() -> None:
            attention(q, k, v, args.causal)

        warmup_count = warmup_for_budget(
            lambda: _no_grad_call(forward_call),
            warmup_ms=args.warmup_ms,
            minimum_repeats=args.warmup,
        )
        reset_phase_peak()
        forward_samples = timed_samples(
            lambda: _no_grad_call(forward_call),
            rep_ms=args.rep_ms,
            minimum_repeats=args.repeats,
        )
        phase_results["forward"] = {
            "status": "passed",
            "samples_ms": forward_samples,
            "warmup_count": warmup_count,
            **summarize(forward_samples),
            "memory": phase_memory(),
        }
    except torch.cuda.OutOfMemoryError as error:
        phase_results["forward"] = {"status": "oom", "error": str(error)}
        torch.cuda.empty_cache()

    # backward 使用同一张保留的图反复测量；每轮把叶子梯度设为 None，避免
    # 梯度累积的加法开销和数值增长。图内保存的 score/probability 正是题目
    # 要求统计的“backward 开始前显存”。
    memory_before_backward = None
    output_finite = None
    gradients_finite = None
    if phase_results["forward"]["status"] == "passed":
        try:
            q.grad = k.grad = v.grad = None
            output = attention(q, k, v, args.causal)
            synchronize()
            memory_before_backward = cuda_memory()
            output_finite = bool(torch.isfinite(output).all().item())
            def backward_call() -> None:
                _backward_once(output, grad_output, q, k, v)

            warmup_count = warmup_for_budget(
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
            gradients_finite = all(
                tensor.grad is not None and bool(torch.isfinite(tensor.grad).all().item())
                for tensor in (q, k, v)
            )
            phase_results["backward"] = {
                "status": "passed",
                "samples_ms": backward_samples,
                "warmup_count": warmup_count,
                **summarize(backward_samples),
                "memory": phase_memory(),
            }
            del output
        except torch.cuda.OutOfMemoryError as error:
            phase_results["backward"] = {"status": "oom", "error": str(error)}
            if "output" in locals():
                del output
            q.grad = k.grad = v.grad = None
            torch.cuda.empty_cache()
    else:
        phase_results["backward"] = {"status": "skipped_after_forward_oom"}

    # 完整阶段每次重新建立 forward 图，再马上 backward。这样它测量的是
    # 用户真正执行一次训练 Attention 的边界，而不是复用上一阶段的图。
    try:
        q.grad = k.grad = v.grad = None
        def end_to_end_call() -> None:
            _forward_backward_once(attention, q, k, v, grad_output, args.causal)

        warmup_count = warmup_for_budget(
            end_to_end_call,
            warmup_ms=args.warmup_ms,
            minimum_repeats=args.warmup,
        )
        reset_phase_peak()
        end_to_end_samples = timed_samples(
            end_to_end_call,
            rep_ms=args.rep_ms,
            minimum_repeats=args.repeats,
        )
        phase_results["forward_backward"] = {
            "status": "passed",
            "samples_ms": end_to_end_samples,
            "warmup_count": warmup_count,
            **summarize(end_to_end_samples),
            "memory": phase_memory(),
        }
    except torch.cuda.OutOfMemoryError as error:
        phase_results["forward_backward"] = {"status": "oom", "error": str(error)}
        q.grad = k.grad = v.grad = None
        torch.cuda.empty_cache()

    # 标准实现至少物化 scores 和 probabilities 两个 B*S*S 张量。这个理论项
    # 不替代 allocator 实测，但能直接解释序列翻倍时为何显存约变成四倍。
    quadratic_tensor_mib = args.batch_size * args.sequence_length**2 * torch.empty((), dtype=dtype).element_size() / MIB
    return {
        "event": "pytorch_attention_benchmark",
        "status": "passed" if all(phase["status"] == "passed" for phase in phase_results.values()) else "partial_oom",
        "implementation": args.implementation,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "head_dimension": args.head_dimension,
        "causal": args.causal,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "compile_cold_start_ms": compile_cold_start_ms,
        "forward": phase_results["forward"],
        "backward": phase_results["backward"],
        "forward_backward": phase_results["forward_backward"],
        "memory_before_backward": memory_before_backward,
        "theoretical_memory": {
            "one_batch_score_or_probability_mib": quadratic_tensor_mib,
            "scores_plus_probabilities_mib": 2 * quadratic_tensor_mib,
        },
        "output_finite": output_finite,
        "gradients_finite": gradients_finite,
        "environment": {
            **hardware_metadata("cuda"),
        },
        "allocator": allocator,
    }


def oom_result(args: argparse.Namespace, error: torch.cuda.OutOfMemoryError) -> dict[str, Any]:
    """OOM 是二次复杂度实验的有效边界，必须落盘而不是吞掉。"""

    return {
        "event": "pytorch_attention_benchmark",
        "status": "oom",
        "implementation": args.implementation,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "head_dimension": args.head_dimension,
        "causal": args.causal,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "allocator_limit_gib": args.allocator_limit_gib,
        "error": str(error),
    }


@dataclass(frozen=True)
class AttentionCase:
    """矩阵中一条可独立恢复的 Attention 配置。"""

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
    """逐 case 启动子进程，保存命令、stdout/stderr 与汇总 JSONL。"""

    run_dir = args.run_dir or REPO_ROOT / "artifacts" / "runs" / datetime.now().astimezone().strftime(
        "%Y%m%d-%H%M%S-attention-matrix"
    )
    # 外层的 A2-K runner 会预先创建这个目录来保存 command/stdout/stderr。
    # 目录存在不代表需要续跑；是否跳过 case 由 results.jsonl 和 --resume 决定。
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    completed_ids: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_ids.add(json.loads(line)["case_id"])

    cases = [
        AttentionCase(implementation, dtype, sequence_length, head_dimension, args.causal)
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
            "cs336_systems.a2k.attention_benchmark",
            "single",
            "--implementation",
            case.implementation,
            "--dtype",
            case.dtype,
            "--sequence-length",
            str(case.sequence_length),
            "--head-dimension",
            str(case.head_dimension),
            "--batch-size",
            str(args.batch_size),
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
            result = {
                "event": "pytorch_attention_benchmark",
                "status": "process_error",
                "stderr_tail": completed.stderr[-4000:],
            }
            process_errors += 1
        result.update(case_id=case.case_id, return_code=completed.returncode)
        with results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(cases)}] {case.case_id}: {result['status']}", flush=True)

    (run_dir / "status.json").write_text(
        json.dumps({"process_errors": process_errors, "case_count": len(cases)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"run_dir={run_dir}")
    return int(process_errors > 0)


def parse_args() -> argparse.Namespace:
    """single 用于独立 case，matrix 用于课程完整笛卡尔积。"""

    parser = argparse.ArgumentParser(description="Benchmark eager and compiled PyTorch attention.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single")
    single.add_argument("--implementation", choices=IMPLEMENTATIONS, default="eager")
    single.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    single.add_argument("--batch-size", type=int, default=8)
    single.add_argument("--sequence-length", type=int, required=True)
    single.add_argument("--head-dimension", type=int, required=True)
    single.add_argument("--causal", dest="causal", action="store_true", default=True)
    single.add_argument("--non-causal", dest="causal", action="store_false")
    single.add_argument("--warmup-ms", type=int, default=100)
    single.add_argument("--rep-ms", type=int, default=300)
    single.add_argument("--warmup", type=int, default=3)
    single.add_argument("--repeats", type=int, default=5)
    single.add_argument("--allocator-limit-gib", type=float, default=None)
    single.add_argument("--seed", type=int, default=2026)
    single.add_argument("--allow-oom", action="store_true")
    single.add_argument("--output", type=Path)

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS))
    matrix.add_argument("--dtypes", nargs="+", choices=("fp32", "bf16"), default=["bf16"])
    matrix.add_argument("--sequence-lengths", nargs="+", type=int, default=list(SEQUENCE_LENGTHS))
    matrix.add_argument("--head-dimensions", nargs="+", type=int, default=list(HEAD_DIMENSIONS))
    matrix.add_argument("--batch-size", type=int, default=1)
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
