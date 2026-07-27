"""A2-P 正式实验编排器。

这个文件只负责实验组织，不重新实现 Transformer 或 Attention。它把 A2-P
要求的四组证据统一成可恢复的结果包：

* End-to-End：small、batch=4、context=512、FP32 的三个训练模式；
* Compute Profiling：两个模型规模、三个 context、六个稳定的 train_step trace；
* Mixed Precision：累加误差、ToyModel dtype 观察和语言模型 benchmark 对照；
* Memory Profiling：XL/context=128、2048 的 forward-only 与完整 train_step。

每个 case 都由独立 Python 进程执行。这样一个 OOM、编译错误或 Nsight
故障不会污染下一个 case 的 CUDA allocator；``--resume`` 只跳过已经在
主 JSONL 中落盘的 case。大型 ``.nsys-rep`` 和 memory snapshot 只保存在
实验目录，不会自动复制到公开提交目录。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_SIZES = ("small", "medium")
PROFILE_CONTEXTS = (256, 512, 1024)


@dataclass(frozen=True)
class BenchmarkCase:
    """一条可独立复现的语言模型 benchmark 配置。"""

    suite: str
    case_id: str
    model_size: str
    context_length: int
    batch_size: int
    mode: str
    dtype: str
    warmup: int
    repeats: int
    annotate_attention: bool = False
    annotate_blocks: bool = False
    memory_snapshot: bool = False


def end_to_end_cases(repeats: int) -> list[BenchmarkCase]:
    """展开 A2-P 任务一的最低矩阵。

    ``train_step`` 有意保留 warmup=0 和 warmup=5 两行，用来观察编译、
    CUDA kernel cache 和 allocator 稳定化前后的差异；其它模式使用题目
    要求的五次 warm-up。
    """

    fixed = {"model_size": "small", "context_length": 512, "batch_size": 4, "dtype": "fp32"}
    return [
        BenchmarkCase("end_to_end", "small-forward-w5", mode="forward", warmup=5, repeats=repeats, **fixed),
        BenchmarkCase(
            "end_to_end", "small-forward_backward-w5", mode="forward_backward", warmup=5, repeats=repeats, **fixed
        ),
        BenchmarkCase("end_to_end", "small-train_step-w0", mode="train_step", warmup=0, repeats=repeats, **fixed),
        BenchmarkCase("end_to_end", "small-train_step-w5", mode="train_step", warmup=5, repeats=repeats, **fixed),
    ]


def profile_cases() -> list[BenchmarkCase]:
    """展开任务二的 2 x 3 个 train_step trace。

    256、512、1024 都是大于 128 的二次幂。batch 固定为 4，使 trace 与
    End-to-End 的训练语义一致；每个 case 只保留一个正式 measurement，
    因为 profiling 的目标是定位 kernel/stage，而不是估计分布。
    """

    return [
        BenchmarkCase(
            "profile",
            f"{model_size}-s{context_length}-train_step",
            model_size=model_size,
            context_length=context_length,
            batch_size=4,
            mode="train_step",
            dtype="fp32",
            warmup=5,
            repeats=1,
            annotate_attention=True,
            annotate_blocks=True,
        )
        for model_size in DEFAULT_MODEL_SIZES
        for context_length in PROFILE_CONTEXTS
    ]


def mixed_precision_cases(repeats: int) -> list[BenchmarkCase]:
    """展开 FP32/BF16 autocast 的相同 shape 对照。

    A2-P 的重点是比较相同工作量的数值、时间和显存，不是只比较 forward
    速度。因此这里同时运行 forward、forward_backward 和完整 train_step。
    """

    fixed = {"model_size": "small", "context_length": 512, "batch_size": 4, "warmup": 5, "repeats": repeats}
    return [
        BenchmarkCase(
            "mixed_precision",
            f"small-{mode}-{dtype}",
            mode=mode,
            dtype=dtype,
            **fixed,
        )
        for mode in ("forward", "forward_backward", "train_step")
        for dtype in ("fp32", "bf16-mixed")
    ]


def memory_cases() -> list[BenchmarkCase]:
    """展开任务四的 XL 两个 context、两个执行边界。

    这里使用 batch=1 和 FP32，符合题面在 4090 上 OOM 时的最小诊断口径；
    如果 2048 失败，失败行仍保留，不能把它改名成更小的配置。
    """

    return [
        BenchmarkCase(
            "memory",
            f"xl-s{context_length}-{mode}-fp32",
            model_size="xl",
            context_length=context_length,
            batch_size=1,
            mode=mode,
            dtype="fp32",
            warmup=2,
            repeats=1,
            memory_snapshot=True,
        )
        for context_length in (128, 2048)
        for mode in ("inference", "train_step")
    ]


def memory_fallback_cases() -> list[BenchmarkCase]:
    """返回题面规定的 XL/2048 OOM 诊断顺序。

    这些 case 不会无条件运行：只有初始 XL/context=2048 出现真正的
    ``status=oom`` 时才执行。这样既不把诊断配置冒充最低要求，也不会在
    2048 已成功时浪费 GPU 时间。
    """

    return [
        BenchmarkCase(
            "memory_fallback",
            f"xl-s1024-{mode}-fp32-fallback",
            model_size="xl",
            context_length=1024,
            batch_size=1,
            mode=mode,
            dtype="fp32",
            warmup=2,
            repeats=1,
            memory_snapshot=True,
        )
        for mode in ("inference", "train_step")
    ] + [
        BenchmarkCase(
            "memory_fallback",
            f"large-s2048-{mode}-fp32-fallback",
            model_size="large",
            context_length=2048,
            batch_size=1,
            mode=mode,
            dtype="fp32",
            warmup=2,
            repeats=1,
            memory_snapshot=True,
        )
        for mode in ("inference", "train_step")
    ]


def read_last_json(path: Path) -> dict[str, Any] | None:
    """读取子进程 JSONL 的最后一行；空文件交给调用方记录 process_error。"""

    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")
        output.flush()


def benchmark_command(case: BenchmarkCase, output: Path, *, python: str) -> list[str]:
    """生成单个 benchmark 子进程命令，不通过 shell 拼接参数。"""

    command = [
        python,
        "-m",
        "profiling.benchmark",
        "--device",
        "cuda",
        "--model-size",
        case.model_size,
        "--context-length",
        str(case.context_length),
        "--batch-size",
        str(case.batch_size),
        "--dtype",
        case.dtype,
        "--mode",
        case.mode,
        "--warmup",
        str(case.warmup),
        "--repeats",
        str(case.repeats),
        "--allow-oom",
        "--output",
        str(output),
    ]
    if case.annotate_attention:
        command.append("--annotate-attention")
    if case.annotate_blocks:
        command.append("--annotate-blocks")
    if case.memory_snapshot:
        command.extend(("--memory-snapshot", str(output.with_name("memory_snapshot.pickle"))))
    return command


def run_command(command: list[str], case_dir: Path, *, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    """保存命令并执行；dry-run 只创建目录和命令文件。"""

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    if dry_run:
        return None
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def run_benchmark_case(case: BenchmarkCase, run_dir: Path, *, python: str, dry_run: bool) -> dict[str, Any]:
    """运行普通 benchmark case，并在失败时保留可诊断记录。"""

    case_dir = run_dir / "cases" / case.suite / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / "result.jsonl"
    command = benchmark_command(case, output, python=python)
    completed = run_command(command, case_dir, dry_run=dry_run)
    if dry_run:
        return {"case_id": case.case_id, "suite": case.suite, "status": "dry_run", "command": command}
    result = read_last_json(output)
    if result is None:
        result = {
            "event": "benchmark_training_step",
            "status": "process_error",
            "error": (completed.stderr if completed else "")[-4000:],
        }
    result.update(case_id=case.case_id, suite=case.suite, return_code=completed.returncode if completed else None)
    return result


def nsys_profile_command(base_command: list[str], case_dir: Path) -> tuple[list[str] | None, Path]:
    """把 benchmark 命令包进 Nsight Systems，并返回 trace 基名。"""

    nsys = shutil.which("nsys")
    trace_base = case_dir / "trace"
    if nsys is None:
        return None, trace_base
    return [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        # 集群的 Nsight Systems 2024.3 不支持旧版的 --pytorch 参数；
        # benchmark.py 自己写入的 NVTX 阶段仍会保留在 trace 中。
        "--output",
        str(trace_base),
        # 该集群版本会把独立的 `--` 当成自己的模糊选项前缀，
        # 因此直接接应用命令；这一形式已由 nsys_matrix.sbatch 验证。
        *base_command,
    ], trace_base


def normalize_nsys_column(column: str) -> str:
    """把 ``Total Time (ns)`` 等版本相关表头归一成稳定键名。"""

    return re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")


def nsys_time_ms(value: str, column: str) -> str:
    """把 Nsight 时间字段统一转换成毫秒字符串。"""

    if not value.strip():
        return ""
    number = float(value)
    normalized = normalize_nsys_column(column)
    if normalized.endswith("_ns"):
        number /= 1_000_000
    elif normalized.endswith("_us"):
        number /= 1_000
    elif normalized.endswith("_s") and not normalized.endswith("_ms"):
        number *= 1_000
    return f"{number:.6f}"


def parse_nsys_kernel_rows(text: str, *, case: BenchmarkCase) -> list[dict[str, Any]]:
    """尽量从 ``nsys stats --format csv`` 提取 kernel 行。

    不同 Nsight 版本的列名略有差异，因此解析器只依赖常见的 Name、Calls
    和 Total Time 列。原始 stats 输出仍完整保留，解析失败不会丢失 trace。
    """

    lines = [line for line in text.splitlines() if line.count(",") >= 2]
    header_index = next((index for index, line in enumerate(lines) if "Name" in line and "Time" in line), None)
    if header_index is None:
        return []
    table = list(csv.reader(lines[header_index:]))
    if not table:
        return []
    header = table[0]
    normalized = [normalize_nsys_column(column) for column in header]

    def find_column(*names: str) -> int | None:
        for name in names:
            if name in normalized:
                return normalized.index(name)
        return None

    name_index = find_column("name", "kernel_name")
    calls_index = find_column("instances", "num_calls", "calls", "invocations")
    total_index = find_column("total_time_ns", "total_time_us", "total_time_ms", "total_time", "time_ns")
    if name_index is None:
        return []

    rows: list[dict[str, Any]] = []
    for row in table[1:11]:
        if len(row) <= name_index or not row[name_index].strip():
            continue
        rows.append(
            {
                "case_id": case.case_id,
                "model_size": case.model_size,
                "context_length": case.context_length,
                "mode": case.mode,
                "kind": "kernel",
                "stage": "",
                "op_or_kernel": row[name_index].strip(),
                "calls": row[calls_index].strip() if calls_index is not None and len(row) > calls_index else "",
                "total_cpu_time": "",
                "total_cuda_time": nsys_time_ms(row[total_index], header[total_index])
                if total_index is not None and len(row) > total_index
                else "",
            }
        )
    return rows


def parse_nsys_report_rows(text: str, *, case: BenchmarkCase, report: str) -> list[dict[str, Any]]:
    """解析单个 Nsight report，并统一成轻量 trace_summary 行。"""

    if report == "cuda_gpu_kern_sum":
        return parse_nsys_kernel_rows(text, case=case)

    lines = [line for line in text.splitlines() if line.count(",") >= 2]
    header_index = next(
        (index for index, line in enumerate(lines) if ("Name" in line or "Range" in line) and "Time" in line),
        None,
    )
    if header_index is None:
        return []
    table = list(csv.reader(lines[header_index:]))
    if not table:
        return []
    header = table[0]
    normalized = [normalize_nsys_column(column) for column in header]

    def find_column(*names: str) -> int | None:
        for name in names:
            if name in normalized:
                return normalized.index(name)
        return None

    name_index = find_column("name", "range", "range_name", "nvtx_range")
    calls_index = find_column("instances", "num_calls", "calls", "invocations")
    total_index = find_column("total_time_ns", "total_time_us", "total_time_ms", "total_time", "time_ns")
    if name_index is None:
        return []
    kind = "stage" if report == "nvtxpp_sum" else "cuda_api"
    rows: list[dict[str, Any]] = []
    for row in table[1:31]:
        if len(row) <= name_index or not row[name_index].strip():
            continue
        name = row[name_index].strip()
        rows.append(
            {
                "case_id": case.case_id,
                "model_size": case.model_size,
                "context_length": case.context_length,
                "mode": case.mode,
                "kind": kind,
                "stage": name if kind == "stage" else "",
                "op_or_kernel": name,
                "calls": row[calls_index].strip() if calls_index is not None and len(row) > calls_index else "",
                "total_cpu_time": nsys_time_ms(row[total_index], header[total_index])
                if kind == "cuda_api" and total_index is not None and len(row) > total_index
                else "",
                "total_cuda_time": nsys_time_ms(row[total_index], header[total_index])
                if kind == "stage" and total_index is not None and len(row) > total_index
                else "",
            }
        )
    return rows


def run_profile_case(case: BenchmarkCase, run_dir: Path, *, python: str, dry_run: bool) -> dict[str, Any]:
    """运行一个 Nsight case；没有 nsys 时明确写 tool_unavailable。"""

    case_dir = run_dir / "cases" / case.suite / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / "result.jsonl"
    base_command = benchmark_command(case, output, python=python)
    profile_command, trace_base = nsys_profile_command(base_command, case_dir)
    if profile_command is None:
        (case_dir / "command.txt").write_text(subprocess.list2cmdline(base_command) + "\n", encoding="utf-8")
        if dry_run:
            return {
                "case_id": case.case_id,
                "suite": case.suite,
                "status": "dry_run",
                "tool": "nsys",
                "command": ["nsys", "profile", "...", *base_command],
            }
        return {
            "case_id": case.case_id,
            "suite": case.suite,
            "status": "tool_unavailable",
            "tool": "nsys",
            "error": "nsys executable was not found on PATH",
        }
    completed = run_command(profile_command, case_dir, dry_run=dry_run)
    if dry_run:
        return {"case_id": case.case_id, "suite": case.suite, "status": "dry_run", "command": profile_command}
    result = read_last_json(output)
    if result is None:
        result = {
            "event": "benchmark_training_step",
            "status": "process_error",
            "error": (completed.stderr if completed else "")[-4000:],
        }
    result.update(
        case_id=case.case_id,
        suite=case.suite,
        tool="nsys",
        trace_file=str(trace_base.with_suffix(".nsys-rep").name),
        return_code=completed.returncode if completed else None,
    )

    trace_path = trace_base.with_suffix(".nsys-rep")
    summary_rows: list[dict[str, Any]] = []
    nsys = shutil.which("nsys")
    if completed and completed.returncode == 0 and nsys and trace_path.exists():
        for report in ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtxpp_sum"):
            stats_command = [nsys, "stats", "--report", report, "--format", "csv", str(trace_path)]
            stats = subprocess.run(stats_command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            (case_dir / f"nsys_{report}.csv").write_text(stats.stdout, encoding="utf-8")
            (case_dir / f"nsys_{report}.stderr.log").write_text(stats.stderr, encoding="utf-8")
            summary_rows.extend(parse_nsys_report_rows(stats.stdout, case=case, report=report))
    result["summary_rows"] = summary_rows
    result["kernel_summary_rows"] = len([row for row in summary_rows if row.get("kind") == "kernel"])
    write_json(case_dir / "result.json", result)
    write_json(case_dir / "profile_metadata.json", {"command": profile_command, "trace_file": str(trace_path.name)})
    return result


def benchmark_row(record: dict[str, Any]) -> dict[str, Any]:
    timing = record.get("timing") or {}
    peak = record.get("peak_memory") or {}
    model = record.get("model") or {}
    return {
        "case_id": record.get("case_id"),
        "suite": record.get("suite"),
        "status": record.get("status"),
        "model_size": record.get("model_size"),
        "batch_size": record.get("batch_size"),
        "context_length": model.get("context_length"),
        "mode": record.get("mode"),
        "precision": record.get("precision"),
        "warmup": record.get("warmup"),
        "repeats": record.get("repeats"),
        "mean_ms": timing.get("mean_ms"),
        "std_ms": timing.get("std_ms"),
        "cv": timing.get("cv"),
        "p20_ms": timing.get("p20_ms"),
        "p50_ms": timing.get("p50_ms"),
        "p80_ms": timing.get("p80_ms"),
        "measurement_count": timing.get("measurement_count"),
        "step_time_ms_samples": json.dumps(timing.get("samples_ms"), ensure_ascii=False),
        "peak_allocated_mib": peak.get("max_allocated_mib"),
        "peak_reserved_mib": peak.get("max_reserved_mib"),
        "gpu": (record.get("environment") or {}).get("gpu_name"),
        "error": record.get("error"),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["case_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_accumulation(run_dir: Path, *, python: str, dry_run: bool) -> dict[str, Any]:
    """运行四种累加写法；它不需要 CUDA，但仍保存命令和 raw JSON。"""

    case_dir = run_dir / "cases" / "mixed_precision" / "accumulation"
    output = case_dir / "accumulation.jsonl"
    # 该脚本把 --output 注册在顶层 parser；argparse 要求它位于子命令之前。
    command = [python, "-m", "profiling.mixed_precision", "--output", str(output), "accumulation"]
    completed = run_command(command, case_dir, dry_run=dry_run)
    if dry_run:
        return {"case_id": "accumulation", "status": "dry_run", "command": command}
    result = read_last_json(output)
    if result is None:
        result = {"event": "mixed_precision_accumulation", "status": "process_error", "error": completed.stderr[-4000:]}
    result["case_id"] = "accumulation"
    return result


def run_toy_dtypes(run_dir: Path, *, python: str, dry_run: bool) -> dict[str, Any]:
    """运行 CUDA BF16 ToyModel dtype 观察。"""

    case_dir = run_dir / "cases" / "mixed_precision" / "toy-bf16"
    output = case_dir / "toy.jsonl"
    command = [
        python,
        "-m",
        "profiling.mixed_precision",
        "--output",
        str(output),
        "toy-dtypes",
        "--dtype",
        "bf16",
    ]
    completed = run_command(command, case_dir, dry_run=dry_run)
    if dry_run:
        return {"case_id": "toy-bf16", "status": "dry_run", "command": command}
    result = read_last_json(output)
    if result is None:
        result = {"event": "mixed_precision_toy_dtypes", "status": "process_error", "error": completed.stderr[-4000:]}
    result["case_id"] = "toy-bf16"
    return result


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["case_id"]] = record
    return records


def run_suite(args: argparse.Namespace) -> int:
    run_dir = args.run_dir or REPO_ROOT / "artifacts" / "a2p-formal"
    # 外层 Slurm/formal runner 可能已经创建 run_dir；续跑逻辑不应阻止首次写入。
    run_dir.mkdir(parents=True, exist_ok=True)
    python = args.python
    all_records: list[dict[str, Any]] = []
    process_errors = 0

    suites: list[str] = [args.suite] if args.suite != "all" else ["end_to_end", "profile", "mixed_precision", "memory"]
    suite_cases: dict[str, list[BenchmarkCase]] = {
        "end_to_end": end_to_end_cases(args.repeats),
        "profile": profile_cases(),
        "mixed_precision": mixed_precision_cases(args.repeats),
        "memory": memory_cases(),
    }
    manifest = {
        "assignment": "A2-P",
        "created_at": datetime.now().astimezone().isoformat(),
        "python": python,
        "repeats": args.repeats,
        "suites": suites,
        "cases": [asdict(case) for suite in suites for case in suite_cases[suite]],
        "profile_tool": "nsys",
        "large_files_policy": "trace and memory snapshots remain in artifacts and are not copied to submission",
    }
    write_json(run_dir / "manifest.json", manifest)

    for suite in suites:
        results_path = run_dir / f"{suite}.jsonl"
        existing = load_existing(results_path) if args.resume else {}
        records: list[dict[str, Any]] = []
        for case in suite_cases[suite]:
            if case.case_id in existing:
                record = existing[case.case_id]
            elif suite == "profile":
                record = run_profile_case(case, run_dir, python=python, dry_run=args.dry_run)
                if not args.dry_run:
                    append_jsonl(results_path, record)
            else:
                record = run_benchmark_case(case, run_dir, python=python, dry_run=args.dry_run)
                if not args.dry_run:
                    append_jsonl(results_path, record)
            records.append(record)
            all_records.append(record)
            process_errors += record.get("status") == "process_error"

        if suite == "memory" and not args.dry_run:
            initial_2048_oom = any(
                record.get("status") == "oom"
                and record.get("model_size") == "xl"
                and (record.get("model") or {}).get("context_length") == 2048
                for record in records
            )
            if initial_2048_oom:
                fallback_path = run_dir / "memory_fallback.jsonl"
                fallback_existing = load_existing(fallback_path) if args.resume else {}
                manifest["memory_fallback_triggered"] = True
                manifest["memory_fallback_cases"] = [asdict(case) for case in memory_fallback_cases()]
                write_json(run_dir / "manifest.json", manifest)
                for case in memory_fallback_cases():
                    record = fallback_existing.get(case.case_id) or run_benchmark_case(
                        case, run_dir, python=python, dry_run=args.dry_run
                    )
                    if case.case_id not in fallback_existing:
                        append_jsonl(fallback_path, record)
                    records.append(record)
                    all_records.append(record)
                    process_errors += record.get("status") == "process_error"

        if suite in ("end_to_end", "mixed_precision") and not args.dry_run:
            write_rows(run_dir / "benchmark.csv", [benchmark_row(record) for record in all_records if record.get("event") == "benchmark_training_step"])

    if "mixed_precision" in suites:
        auxiliary = [
            run_accumulation(run_dir, python=python, dry_run=args.dry_run),
            run_toy_dtypes(run_dir, python=python, dry_run=args.dry_run),
        ]
        if not args.dry_run:
            write_json(run_dir / "mixed_precision.json", {"experiments": auxiliary})
        all_records.extend(auxiliary)

    if "profile" in suites and not args.dry_run:
        profile_rows: list[dict[str, Any]] = []
        for record in all_records:
            if record.get("suite") != "profile":
                continue
            common = {
                "case_id": record.get("case_id"),
                "model_size": record.get("model_size"),
                "context_length": (record.get("model") or {}).get("context_length"),
                "mode": record.get("mode"),
                "dtype": record.get("precision"),
                "tool": record.get("tool", "nsys"),
                "status": record.get("status"),
                "trace_file": record.get("trace_file"),
                "command": " ".join(record.get("command", [])) if isinstance(record.get("command"), list) else "",
                "error": record.get("error"),
            }
            rows = record.get("summary_rows") or []
            if not rows:
                profile_rows.append({
                    **common,
                    "kind": "case_summary",
                    "stage": "profile/warmup;profile/measure;forward;backward;optimizer;attention/scores;attention/softmax;attention/value",
                    "op_or_kernel": "",
                    "calls": "",
                    "total_cpu_time": "",
                    "total_cuda_time": "",
                })
            else:
                profile_rows.extend({**common, **row} for row in rows)
        write_rows(run_dir / "profile" / "trace_summary.csv", profile_rows)
        write_json(run_dir / "profile" / "run_metadata.json", {"tool": "nsys", "cases": profile_rows})

    if "memory" in suites and not args.dry_run:
        memory_rows = [benchmark_row(record) for record in all_records if record.get("suite") == "memory"]
        write_rows(run_dir / "memory" / "peaks.csv", memory_rows)
        write_json(run_dir / "memory" / "run_metadata.json", {"cases": memory_rows, "snapshot_policy": "one snapshot per case"})

    write_json(
        run_dir / "run_metadata.json",
        {
            "assignment": "A2-P",
            "created_at": datetime.now().astimezone().isoformat(),
            "python": python,
            "suites": suites,
            "dry_run": args.dry_run,
            "case_count": len(all_records),
            "process_errors": process_errors,
            "note": "GPU and Nsight pass/fail claims require execution on the 4090 server.",
        },
    )
    print(f"run_dir={run_dir}")
    return 0 if process_errors == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal A2-P experiment package.")
    parser.add_argument("--suite", choices=("all", "end_to_end", "profile", "mixed_precision", "memory"), default="all")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run_suite(parse_args()))
