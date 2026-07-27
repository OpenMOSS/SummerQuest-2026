"""A2-K 正式实验总入口。

这个运行器只负责组织实验，不重新实现 Attention 或 FlashAttention 数学。
每一类实验都放在独立目录，并且把实际执行命令、标准输出、标准错误和
逐条 JSONL 结果保存下来。这样某个长序列 OOM 或某次编译失败时，可以从
其它 case 继续，而不会把失败配置从最终数据中删除。

正式流水线包含四部分：

* checkpoint：调用 ``a2k_checkpoint`` 扫描 medium/24-layer 的分组策略；
* attention：显式 ``QK^T -> causal mask -> softmax -> PV`` 与 compiled 对照；
* correctness：三种随机种子、三种 head dimension、causal/non-causal，保存
  ``O/L/dQ/dK/dV`` 的实际 tensor 文件；
* flash：eager PyTorch、compiled PyTorch、student Triton 的性能矩阵，另外
  保留 16384 长序列的 eager/Triton 边界。

本地 CPU 可以执行 ``--dry-run`` 和 correctness 的结构检查，但 Triton/GPU
通过与正式延迟数字只能在 4090 计算节点上确认。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from cs336_systems.a2k.experiment_utils import hardware_metadata
from cs336_systems.a2k.flash_attention import (
    PyTorchFlashAttention,
    TRITON_AVAILABLE,
    TritonFlashAttention,
)
from cs336_systems.a2k.result_tables import attention_row, benchmark_row, flash_row


REPO_ROOT = Path(__file__).resolve().parents[2]
ATTENTION_SEQUENCES = (512, 2048, 8192)
ATTENTION_HEAD_DIMS = (64, 128)
FLASH_SEQUENCES = (512, 2048, 8192)
FLASH_BOUNDARY_SEQUENCE = 16384
FLASH_HEAD_DIMS = (64, 128)
MODEL_COMPILE_MODES = ("forward", "forward_backward", "train_step")
CORRECTNESS_SEEDS = (3, 17, 29)
CORRECTNESS_HEAD_DIMS = (32, 64, 128)
CORRECTNESS_SEQUENCE = 128
CORRECTNESS_RELATIVE_EPSILON = 1e-8


@dataclass(frozen=True)
class CommandSpec:
    """一条会被写入 manifest 的外部命令。"""

    name: str
    command: list[str]
    run_dir: Path


def write_json(path: Path, value: Any) -> None:
    """用 UTF-8 写入可审计的 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_command_spec(spec: CommandSpec) -> None:
    """保存命令文本，便于之后在服务器上逐条复现实验。"""

    spec.run_dir.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "command.txt").write_text(subprocess.list2cmdline(spec.command) + "\n", encoding="utf-8")


def run_logged(spec: CommandSpec, *, dry_run: bool) -> int:
    """运行一条矩阵子命令，并保留 stdout/stderr。"""

    write_command_spec(spec)
    if dry_run:
        print(f"[dry-run] {spec.name}: {subprocess.list2cmdline(spec.command)}", flush=True)
        return 0
    completed = subprocess.run(spec.command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    (spec.run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (spec.run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    write_json(spec.run_dir / "status.json", {"name": spec.name, "return_code": completed.returncode})
    return completed.returncode


def python_module_command(python: str, module: str, *arguments: str) -> list[str]:
    """生成不经过 shell 的 Python module 命令。"""

    return [python, "-m", module, *arguments]


def run_checkpoint(args: argparse.Namespace, root: Path) -> int:
    """调用已有 checkpoint runner。"""

    run_dir = root / "checkpoint"
    command = python_module_command(
        args.python,
        "cs336_systems.a2k.checkpoint_benchmark",
        "--run-dir",
        str(run_dir),
        "--python",
        args.python,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    )
    if args.resume:
        command.append("--resume")
    return run_logged(CommandSpec("checkpoint", command, run_dir), dry_run=args.dry_run)


def run_attention(args: argparse.Namespace, root: Path) -> int:
    """运行 A2-K 显式 Attention 与 torch.compile 矩阵。"""

    run_dir = root / "attention"
    command = python_module_command(
        args.python,
        "cs336_systems.a2k.attention_benchmark",
        "matrix",
        "--run-dir",
        str(run_dir),
        "--implementations",
        "eager",
        "compiled",
        "--dtypes",
        "bf16",
        "--sequence-lengths",
        *(str(value) for value in ATTENTION_SEQUENCES),
        "--head-dimensions",
        *(str(value) for value in ATTENTION_HEAD_DIMS),
        "--batch-size",
        "1",
        "--causal",
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--allocator-limit-gib",
        "23",
    )
    if args.resume:
        command.append("--resume")
    code = run_logged(CommandSpec("attention", command, run_dir), dry_run=args.dry_run)
    if code == 0 and not args.dry_run:
        records = read_jsonl(run_dir / "results.jsonl")
        write_rows(run_dir.parent / "attention_baseline.csv", [attention_row(record) for record in records])
    model_compile_code = run_model_compile(args, root)
    return int(code != 0 or model_compile_code != 0)


def model_compile_command(
    args: argparse.Namespace,
    output: Path,
    *,
    mode: str,
    compiled: bool,
) -> list[str]:
    """生成 Stanford small 模型级 compile 对照命令。"""

    command = python_module_command(
        args.python,
        "cs336_systems.a2k.model_benchmark",
        "--device",
        "cuda",
        "--model-size",
        "small",
        "--context-length",
        "512",
        "--batch-size",
        "1",
        "--dtype",
        "bf16-mixed",
        "--mode",
        mode,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--allocator-limit-gib",
        "23",
        "--allow-oom",
        "--output",
        str(output),
    )
    if compiled:
        command.append("--compile-model")
    return command


def run_model_compile(args: argparse.Namespace, root: Path) -> int:
    """运行整个 Stanford small Transformer 的 eager/compiled 对照。"""

    run_dir = root / "model_compile"
    run_dir.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, str, bool]] = [
        (f"small-{mode}-{'compiled' if compiled else 'eager'}", mode, compiled)
        for mode in MODEL_COMPILE_MODES
        for compiled in (False, True)
    ]
    commands: list[CommandSpec] = []
    for case_id, mode, compiled in cases:
        case_dir = run_dir / "cases" / case_id
        output = case_dir / "result.jsonl"
        commands.append(
            CommandSpec(
                f"model_compile_{case_id}",
                model_compile_command(args, output, mode=mode, compiled=compiled),
                case_dir,
            )
        )

    codes: list[int] = []
    for spec in commands:
        result_path = spec.run_dir / "result.jsonl"
        if args.resume and result_path.exists() and result_path.stat().st_size > 0:
            print(f"[resume] {spec.name}", flush=True)
            codes.append(0)
        else:
            codes.append(run_logged(spec, dry_run=args.dry_run))
    if not args.dry_run:
        records: list[dict[str, Any]] = []
        for (case_id, _mode, _compiled), spec, code in zip(cases, commands, codes, strict=True):
            result = read_last_json(spec.run_dir / "result.jsonl")
            if result is None:
                result = {
                    "event": "benchmark_training_step",
                    "status": "process_error",
                    "case_id": case_id,
                    "error": (spec.run_dir / "stderr.log").read_text(encoding="utf-8")[-4000:]
                    if (spec.run_dir / "stderr.log").exists()
                    else "",
                }
            result.update(case_id=case_id, return_code=code)
            records.append(result)
        write_json(root / "compile_comparison.jsonl", records)
        write_rows(root / "compile_comparison.csv", [benchmark_row(record) for record in records])
    return int(any(code != 0 for code in codes))


def flash_command(
    args: argparse.Namespace,
    run_dir: Path,
    implementations: tuple[str, ...],
    sequences: tuple[int, ...],
) -> list[str]:
    """生成一组 FlashAttention benchmark 命令。"""

    command = python_module_command(
        args.python,
        "cs336_systems.a2k.flash_benchmark",
        "matrix",
        "--run-dir",
        str(run_dir),
        "--implementations",
        *implementations,
        "--dtypes",
        "bf16",
        "--sequence-lengths",
        *(str(value) for value in sequences),
        "--head-dimensions",
        *(str(value) for value in FLASH_HEAD_DIMS),
        "--causal",
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--allocator-limit-gib",
        "23",
    )
    if args.resume:
        command.append("--resume")
    return command


def run_flash(args: argparse.Namespace, root: Path) -> int:
    """运行核心矩阵和 16384 长序列边界，并合并轻量 CSV。"""

    core_dir = root / "flash" / "core"
    boundary_dir = root / "flash" / "boundary"
    specs = (
        CommandSpec(
            "flash_core",
            flash_command(args, core_dir, ("pytorch", "compiled", "triton"), FLASH_SEQUENCES),
            core_dir,
        ),
        CommandSpec(
            "flash_boundary",
            flash_command(args, boundary_dir, ("pytorch", "triton"), (FLASH_BOUNDARY_SEQUENCE,)),
            boundary_dir,
        ),
    )
    codes = [run_logged(spec, dry_run=args.dry_run) for spec in specs]
    if not args.dry_run and all(code == 0 for code in codes):
        records = read_jsonl(core_dir / "results.jsonl") + read_jsonl(boundary_dir / "results.jsonl")
        write_json(root / "flash_benchmark.jsonl", records)
        write_rows(root / "flash_benchmark.csv", [flash_row(record) for record in records])
    return int(any(code != 0 for code in codes))


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回标准 Attention 的 O 与 L，L 是未归一化 score 的 log-sum-exp。"""

    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    if causal:
        query_index = torch.arange(q.shape[-2], device=q.device)[:, None]
        key_index = torch.arange(k.shape[-2], device=k.device)[None, :]
        scores = torch.where(query_index >= key_index, scores, torch.full_like(scores, -1e6))
    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ v, torch.logsumexp(scores, dim=-1)


def saved_lse(output: torch.Tensor, expected_shape: tuple[int, int]) -> torch.Tensor:
    """从 autograd 保存对象中提取唯一的 L。"""

    saved = [tensor for tensor in output.grad_fn.saved_tensors if tuple(tensor.shape) == expected_shape]
    if len(saved) != 1:
        raise RuntimeError(f"expected one saved L tensor with shape {expected_shape}, found {len(saved)}")
    return saved[0]


def correctness_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """返回扩展正确性矩阵使用的绝对和相对容差。"""

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    return tolerance, tolerance


def tensor_error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """计算可落盘的最大绝对误差和最大相对误差。

    相对误差在参考值接近零时会天然放大，因此它只作为诊断字段；最终
    pass/fail 仍使用同时包含 ``atol`` 与 ``rtol`` 的 ``torch.allclose``。
    """

    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(CORRECTNESS_RELATIVE_EPSILON)
    return {
        "max_abs_error": float(difference.max().item()),
        "max_rel_error": float((difference / denominator).max().item()),
    }


def run_correctness_case(
    *,
    seed: int,
    head_dimension: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    case_dir: Path,
) -> dict[str, Any]:
    """比较 reference、PyTorch Flash 和 Triton Flash 的五类张量。"""

    case_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    shape = (1, CORRECTNESS_SEQUENCE, head_dimension)
    base_q = torch.randn(shape, device=device, dtype=dtype)
    base_k = torch.randn(shape, device=device, dtype=dtype)
    base_v = torch.randn(shape, device=device, dtype=dtype)
    grad_output = torch.randn(shape, device=device, dtype=dtype)

    reference_q = base_q.detach().clone().requires_grad_(True)
    reference_k = base_k.detach().clone().requires_grad_(True)
    reference_v = base_v.detach().clone().requires_grad_(True)
    reference_output, reference_lse = reference_attention(reference_q, reference_k, reference_v, causal)
    reference_output.backward(grad_output)
    reference_values = {
        "O": reference_output.detach().clone(),
        "L": reference_lse.detach().clone(),
        "dQ": reference_q.grad.detach().clone(),
        "dK": reference_k.grad.detach().clone(),
        "dV": reference_v.grad.detach().clone(),
    }
    torch.save(reference_values, case_dir / "reference.pt")

    record: dict[str, Any] = {
        "event": "a2k_flash_correctness",
        "status": "passed",
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "batch_size": 1,
        "sequence_length": CORRECTNESS_SEQUENCE,
        "head_dimension": head_dimension,
        "causal": causal,
        "implementations": {},
        "environment": hardware_metadata(device),
    }

    for name, function in (("pytorch", PyTorchFlashAttention), ("triton", TritonFlashAttention)):
        if name == "triton" and (not device.type == "cuda" or not TRITON_AVAILABLE):
            record["implementations"][name] = {"status": "tool_unavailable"}
            continue
        try:
            q = base_q.detach().clone().requires_grad_(True)
            k = base_k.detach().clone().requires_grad_(True)
            v = base_v.detach().clone().requires_grad_(True)
            output = function.apply(q, k, v, causal)
            lse = saved_lse(output, (1, CORRECTNESS_SEQUENCE))
            output.backward(grad_output)
            values = {
                "O": output.detach().clone(),
                "L": lse.detach().clone(),
                "dQ": q.grad.detach().clone(),
                "dK": k.grad.detach().clone(),
                "dV": v.grad.detach().clone(),
            }
            tensor_path = case_dir / f"{name}.pt"
            torch.save(values, tensor_path)
            atol, rtol = correctness_tolerances(dtype)
            comparisons = {
                key: tensor_error_metrics(values[key], reference_values[key])
                for key in ("O", "L", "dQ", "dK", "dV")
            }
            tensor_pass = {
                key: bool(torch.allclose(values[key].float(), reference_values[key].float(), atol=atol, rtol=rtol))
                for key in ("O", "L", "dQ", "dK", "dV")
            }
            all_finite = all(bool(torch.isfinite(value).all().item()) for value in values.values())
            implementation_passed = all(tensor_pass.values()) and all_finite
            record["implementations"][name] = {
                "status": "passed" if implementation_passed else "failed",
                "tensor_path": str(tensor_path.relative_to(case_dir.parent.parent)),
                "errors": comparisons,
                "tensor_pass": tensor_pass,
                "tolerance": {"atol": atol, "rtol": rtol},
                "all_finite": all_finite,
            }
            if not implementation_passed:
                record["status"] = "failed"
        except torch.cuda.OutOfMemoryError as error:
            record["status"] = "partial_oom"
            record["implementations"][name] = {"status": "oom", "error": str(error)}
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:  # 每个实现单独落盘，保留其它 case 的诊断价值。
            record["status"] = "failed"
            record["implementations"][name] = {"status": "error", "error": repr(error)}
    return record


def run_correctness(args: argparse.Namespace, root: Path) -> int:
    """执行小规模 correctness matrix；GPU 不可用时写清楚 skip。"""

    run_dir = root / "correctness"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seeds": list(CORRECTNESS_SEEDS),
        "head_dimensions": list(CORRECTNESS_HEAD_DIMS),
        "causal": [False, True],
        "dtype": "fp32",
        "sequence_length": CORRECTNESS_SEQUENCE,
        "tolerance": {"atol": 1e-2, "rtol": 1e-2},
    }
    write_json(run_dir / "manifest.json", manifest)
    results_path = run_dir / "correctness.jsonl"
    if args.dry_run:
        print(f"[dry-run] correctness cases: {len(CORRECTNESS_SEEDS) * len(CORRECTNESS_HEAD_DIMS) * 2}")
        return 0

    records: list[dict[str, Any]] = []
    existing = {record["case_id"]: record for record in read_jsonl(results_path)} if args.resume else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in CORRECTNESS_SEEDS:
        for head_dimension in CORRECTNESS_HEAD_DIMS:
            for causal in (False, True):
                case_id = f"seed{seed}-d{head_dimension}-{'causal' if causal else 'noncausal'}"
                if case_id in existing:
                    records.append(existing[case_id])
                    print(f"correctness {case_id}: resume", flush=True)
                    continue
                record = run_correctness_case(
                    seed=seed,
                    head_dimension=head_dimension,
                    causal=causal,
                    dtype=torch.float32,
                    device=device,
                    case_dir=run_dir / "cases" / case_id,
                )
                record["case_id"] = case_id
                if not TRITON_AVAILABLE or device.type != "cuda":
                    record["status"] = "partial_tool_unavailable"
                    record["unavailable_reason"] = "Triton correctness requires a CUDA environment"
                records.append(record)
                print(f"correctness {case_id}: {record['status']}", flush=True)
    results_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return int(any(record["status"] == "failed" for record in records))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取矩阵结果；不存在时返回空列表，便于 dry-run 之外的错误诊断。"""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_last_json(path: Path) -> dict[str, Any] | None:
    """读取 benchmark 子进程 JSONL 的最后一条完整记录。"""

    records = read_jsonl(path)
    return records[-1] if records else None


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """把扁平结果写成 CSV，不丢弃 OOM 或 process_error 行。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_suite(args: argparse.Namespace) -> int:
    """按 suite 运行一个或多个模块。"""

    root = args.run_dir or REPO_ROOT / "artifacts" / "a2k-formal"
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "run_metadata.json",
        {
            "assignment": "A2-K",
            "created_at": datetime.now().astimezone().isoformat(),
            "python": args.python,
            "suite": args.suite,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "allocator_limit_gib": 23,
            "dry_run": args.dry_run,
        },
    )
    write_json(
        root / "manifest.json",
        {
            "assignment": "A2-K",
            "suite": args.suite,
            "case_counts": {
                "checkpoint_scan": 5,
                "attention_kernel": len(ATTENTION_SEQUENCES) * len(ATTENTION_HEAD_DIMS) * 2,
                "model_compile": len(MODEL_COMPILE_MODES) * 2,
                "flash_core": len(FLASH_SEQUENCES) * len(FLASH_HEAD_DIMS) * 3,
                "flash_boundary": len(FLASH_HEAD_DIMS) * 2,
                "correctness": len(CORRECTNESS_SEEDS) * len(CORRECTNESS_HEAD_DIMS) * 2,
            },
            "attention_kernel": {
                "batch_size": 1,
                "dtype": "bf16",
                "causal": True,
                "sequence_lengths": list(ATTENTION_SEQUENCES),
                "head_dimensions": list(ATTENTION_HEAD_DIMS),
                "implementations": ["eager", "compiled"],
            },
            "model_compile": {
                "model_size": "small",
                "batch_size": 1,
                "context_length": 512,
                "dtype": "bf16-mixed",
                "modes": list(MODEL_COMPILE_MODES),
                "implementations": ["eager", "compiled"],
            },
            "flash": {
                "batch_size": 1,
                "dtype": "bf16",
                "causal": True,
                "core_sequence_lengths": list(FLASH_SEQUENCES),
                "boundary_sequence_length": FLASH_BOUNDARY_SEQUENCE,
                "head_dimensions": list(FLASH_HEAD_DIMS),
                "core_implementations": ["pytorch", "compiled", "triton"],
                "boundary_implementations": ["pytorch", "triton"],
            },
            "correctness": {
                "dtype": "fp32",
                "sequence_length": CORRECTNESS_SEQUENCE,
                "seeds": list(CORRECTNESS_SEEDS),
                "head_dimensions": list(CORRECTNESS_HEAD_DIMS),
                "causal": [False, True],
            },
        },
    )
    selected = ("checkpoint", "attention", "correctness", "flash") if args.suite == "all" else (args.suite,)
    runners = {
        "checkpoint": run_checkpoint,
        "attention": run_attention,
        "correctness": run_correctness,
        "flash": run_flash,
    }
    codes = [runners[name](args, root) for name in selected]
    write_json(root / "status.json", {"suite": args.suite, "return_codes": dict(zip(selected, codes, strict=True))})
    print(f"run_dir={root}")
    return int(any(code != 0 for code in codes))


def parse_args() -> argparse.Namespace:
    """解析正式 runner 参数。"""

    parser = argparse.ArgumentParser(description="Run the formal A2-K experiment suites.")
    parser.add_argument("--suite", choices=("all", "checkpoint", "attention", "correctness", "flash"), default="all")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """程序入口。"""

    args = parse_args()
    if args.warmup_ms < 0 or args.rep_ms <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("rep-ms/repeats must be positive; warmup-ms/warmup may be zero")
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
