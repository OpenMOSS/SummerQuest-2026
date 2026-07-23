from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.config import utc_now, write_json


@dataclass(frozen=True)
class CommandSpec:
    run_id: str
    argv: tuple[str, ...]

    def display(self) -> str:
        return shlex.join(self.argv)


def benchmark_command(
    run_id: str,
    model: str,
    context: int,
    mode: str,
    dtype: str,
    warmup: int,
    steps: int,
    output_root: Path,
) -> CommandSpec:
    output = output_root / "benchmark" / "raw" / f"{run_id.lower()}.json"
    return CommandSpec(
        run_id,
        (
            "uv",
            "run",
            "python",
            "profiling/benchmark.py",
            "--run-id",
            run_id,
            "--model-size",
            model,
            "--batch-size",
            "4",
            "--context-length",
            str(context),
            "--mode",
            mode,
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--dtype",
            dtype,
            "--no-annotate-attention",
            "--output",
            output.as_posix(),
        ),
    )


def preflight_commands(output_root: Path) -> list[CommandSpec]:
    return [
        CommandSpec(
            "PREFLIGHT",
            (
                "uv",
                "run",
                "python",
                "profiling/preflight.py",
                "--device",
                "cuda",
                "--output",
                (output_root / "preflight" / "environment.json").as_posix(),
            ),
        )
    ]


def task1_commands(output_root: Path) -> list[CommandSpec]:
    return [
        benchmark_command("B1", "small", 512, "forward", "fp32", 5, 10, output_root),
        benchmark_command("B2", "small", 512, "forward_backward", "fp32", 5, 10, output_root),
        benchmark_command("B3", "small", 512, "train_step", "fp32", 5, 10, output_root),
        benchmark_command("B4", "small", 512, "train_step", "fp32", 0, 10, output_root),
    ]


def task2_commands(output_root: Path, schedule_policy: str) -> list[CommandSpec]:
    commands = []
    index = 1
    for model in ("small", "medium"):
        for context in (256, 512, 1024):
            run_id = f"P{index}"
            commands.append(
                CommandSpec(
                    run_id,
                    (
                        "uv",
                        "run",
                        "python",
                        "profiling/compute_profile.py",
                        "--run-id",
                        run_id,
                        "--model-size",
                        model,
                        "--batch-size",
                        "4",
                        "--context-length",
                        str(context),
                        "--mode",
                        "train_step",
                        "--warmup",
                        "5",
                        "--steps",
                        "1",
                        "--dtype",
                        "fp32",
                        "--schedule-policy",
                        schedule_policy,
                        "--output-dir",
                        (output_root / "profile" / "raw").as_posix(),
                    ),
                )
            )
            index += 1
    return commands


def task2_smoke_commands(output_root: Path, schedule_policy: str) -> list[CommandSpec]:
    return [
        CommandSpec(
            "P-SMOKE",
            (
                "uv",
                "run",
                "python",
                "profiling/compute_profile.py",
                "--run-id",
                "P-SMOKE",
                "--model-size",
                "tiny",
                "--batch-size",
                "1",
                "--context-length",
                "32",
                "--mode",
                "train_step",
                "--warmup",
                "2",
                "--steps",
                "1",
                "--dtype",
                "fp32",
                "--schedule-policy",
                schedule_policy,
                "--output-dir",
                (output_root / "profile" / "smoke").as_posix(),
            ),
        )
    ]


def mixed_commands(output_root: Path) -> list[CommandSpec]:
    root = output_root / "mixed_precision" / "raw"
    commands = [
        CommandSpec(
            "MP-A",
            (
                "uv",
                "run",
                "python",
                "profiling/mixed_precision.py",
                "accumulation",
                "--run-id",
                "MP-A",
                "--output",
                (root / "accumulation.json").as_posix(),
            ),
        ),
        CommandSpec(
            "MP-T-FP32",
            (
                "uv",
                "run",
                "python",
                "profiling/mixed_precision.py",
                "toy",
                "--run-id",
                "MP-T-FP32",
                "--dtype",
                "fp32",
                "--output",
                (root / "toy_fp32.json").as_posix(),
            ),
        ),
        CommandSpec(
            "MP-T-BF16",
            (
                "uv",
                "run",
                "python",
                "profiling/mixed_precision.py",
                "toy",
                "--run-id",
                "MP-T-BF16",
                "--dtype",
                "bf16",
                "--output",
                (root / "toy_bf16.json").as_posix(),
            ),
        ),
    ]
    index = 1
    for dtype in ("fp32", "bf16"):
        for mode in ("forward", "forward_backward", "train_step"):
            run_id = f"MP-B{index}"
            output = root / f"benchmark_{dtype}_{mode}.json"
            commands.append(
                CommandSpec(
                    run_id,
                    (
                        "uv",
                        "run",
                        "python",
                        "profiling/benchmark.py",
                        "--run-id",
                        run_id,
                        "--model-size",
                        "small",
                        "--batch-size",
                        "4",
                        "--context-length",
                        "512",
                        "--mode",
                        mode,
                        "--warmup",
                        "5",
                        "--steps",
                        "10",
                        "--dtype",
                        dtype,
                        "--no-annotate-attention",
                        "--output",
                        output.as_posix(),
                    ),
                )
            )
            index += 1
    return commands


def memory_commands(output_root: Path) -> list[CommandSpec]:
    output = output_root / "memory" / "raw"
    commands = []
    index = 1
    for context in (128, 2048):
        for mode in ("forward", "train_step"):
            run_id = f"M{index}"
            commands.append(
                CommandSpec(
                    run_id,
                    (
                        "uv",
                        "run",
                        "python",
                        "profiling/memory_snapshot.py",
                        "snapshot",
                        "--run-id",
                        run_id,
                        "--model-size",
                        "xl",
                        "--batch-size",
                        "4",
                        "--context-length",
                        str(context),
                        "--mode",
                        mode,
                        "--warmup",
                        "1",
                        "--steps",
                        "1",
                        "--dtype",
                        "fp32",
                        "--annotate-attention",
                        "--output-dir",
                        output.as_posix(),
                    ),
                )
            )
            index += 1
    commands.append(
        CommandSpec(
            "M-BLOCK",
            (
                "uv",
                "run",
                "python",
                "profiling/memory_snapshot.py",
                "saved-tensors",
                "--run-id",
                "M-BLOCK",
                "--model-size",
                "xl",
                "--batch-size",
                "4",
                "--context-length",
                "128",
                "--dtype",
                "fp32",
                "--output",
                (output / "saved_tensors_xl_ctx128.json").as_posix(),
            ),
        )
    )
    return commands


def memory_fallback_commands(output_root: Path) -> list[CommandSpec]:
    output = output_root / "memory" / "raw"
    commands = []
    index = 1
    for mode, parent_run_id in (("forward", "M3"), ("train_step", "M4")):
        for attempt, (model, context, batch) in enumerate(
            (
                ("xl", 2048, 1),
                ("xl", 1024, 1),
                ("large", 2048, 1),
            ),
            start=1,
        ):
            run_id = f"MF{index}"
            commands.append(
                CommandSpec(
                    run_id,
                    (
                        "uv",
                        "run",
                        "python",
                        "profiling/memory_snapshot.py",
                        "snapshot",
                        "--run-id",
                        run_id,
                        "--model-size",
                        model,
                        "--batch-size",
                        str(batch),
                        "--context-length",
                        str(context),
                        "--mode",
                        mode,
                        "--warmup",
                        "1",
                        "--steps",
                        "1",
                        "--dtype",
                        "fp32",
                        "--annotate-attention",
                        "--requested-model",
                        "xl",
                        "--requested-context",
                        "2048",
                        "--requested-batch",
                        "4",
                        "--fallback-reason",
                        "prior_attempt_oom",
                        "--fallback-parent-run-id",
                        parent_run_id,
                        "--fallback-attempt",
                        str(attempt),
                        "--output-dir",
                        output.as_posix(),
                    ),
                )
            )
            index += 1
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print or execute fixed A2-P experiment matrices")
    parser.add_argument(
        "--suite",
        choices=("preflight", "task1", "task2-smoke", "task2", "mixed", "memory", "memory-fallback", "all"),
        required=True,
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--schedule-policy", choices=("canonical", "visible_warmup"), default="visible_warmup")
    parser.add_argument("--execute", action="store_true", help="without this flag commands are only printed")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def select_commands(args: argparse.Namespace) -> list[CommandSpec]:
    suites = {
        "preflight": preflight_commands(args.results_root),
        "task1": task1_commands(args.results_root),
        "task2-smoke": task2_smoke_commands(args.results_root, args.schedule_policy),
        "task2": task2_commands(args.results_root, args.schedule_policy),
        "mixed": mixed_commands(args.results_root),
        "memory": memory_commands(args.results_root),
        "memory-fallback": memory_fallback_commands(args.results_root),
    }
    if args.suite == "all":
        return suites["preflight"] + suites["task1"] + suites["task2"] + suites["mixed"] + suites["memory"]
    return suites[args.suite]


def execute(commands: Iterable[CommandSpec], continue_on_error: bool) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    final_code = 0
    for command in commands:
        print(f"[{command.run_id}] {command.display()}", flush=True)
        started_at = utc_now()
        result = subprocess.run(command.argv, check=False)
        record: dict[str, object] = {
            "run_id": command.run_id,
            "command": command.display(),
            "started_at": started_at,
            "finished_at": utc_now(),
            "return_code": result.returncode,
        }
        records.append(record)
        if result.returncode != 0:
            final_code = result.returncode
            if not continue_on_error:
                break
    return records, final_code


def update_suite_manifest(path: Path, suite: str, schedule_policy: str, records: list[dict[str, object]]) -> None:
    write_json(
        path,
        {
            "suite": suite,
            "schedule_policy": schedule_policy,
            "records": records,
        },
    )


def main() -> int:
    args = parse_args()
    commands = select_commands(args)
    if args.suite == "memory-fallback" and args.execute:
        raise ValueError("memory-fallback is an ordered candidate list: execute only the next candidate after the prior attempt OOMs")
    if not args.execute:
        for command in commands:
            print(f"[{command.run_id}] {command.display()}")
        return 0

    manifest = args.results_root / "suite_runs" / f"{args.suite}.json"
    records: list[dict[str, object]] = []
    return_code = 0
    for command in commands:
        current, code = execute([command], continue_on_error=False)
        records.extend(current)
        update_suite_manifest(manifest, args.suite, args.schedule_policy, records)
        if code != 0:
            return_code = code
            if not args.continue_on_error:
                break
    print(json.dumps({"manifest": manifest.as_posix(), "return_code": return_code}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
