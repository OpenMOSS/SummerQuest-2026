"""Run the A2-K experiment matrix as independent serial subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "local_results"


def run(command: list[str], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if isinstance(existing, dict) and existing.get("status") in {
            "complete",
            "oom",
            "failed",
        }:
            print(f"skip existing {output}", flush=True)
            return 0 if existing.get("status") == "complete" else 1
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(ROOT), str(ROOT / "cs336-basics"), os.environ.get("PYTHONPATH", "")))
            ),
        },
    )
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-flash", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Do not rerun the extended correctness matrix.",
    )
    args = parser.parse_args()
    results = args.results
    python = sys.executable
    failures: list[str] = []

    if not args.skip_checkpoint:
        for block in (0, 1, 2, 4, 8):
            output = results / "checkpointing" / f"medium_ctx1024_b1_block{block}.json"
            command = [
                python,
                "-m",
                "student_scripts.a2k.checkpoint_benchmark",
                "--model-size",
                "medium",
                "--context-length",
                "1024",
                "--batch-size",
                "1",
                "--block-size",
                str(block),
                "--output",
                str(output),
            ]
            if run(command, output) not in (0, 1):
                failures.append(str(output))
        # The 2048 boundary consists of the required baseline plus the
        # lowest-peak successful 1024 configuration.  Selection is made from
        # completed 1024 JSON files, never from a guessed block size.
        candidates: list[tuple[float, int]] = []
        for block in (0, 1, 2, 4, 8):
            source = results / "checkpointing" / f"medium_ctx1024_b1_block{block}.json"
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("status") == "complete"
                and payload.get("peak_allocated_mib") is not None
            ):
                candidates.append((float(payload["peak_allocated_mib"]), block))
        selected_blocks = {0}
        if candidates:
            selected_blocks.add(min(candidates)[1])
        for block in sorted(selected_blocks):
            output = results / "checkpointing" / f"medium_ctx2048_b1_block{block}.json"
            command = [
                python,
                "-m",
                "student_scripts.a2k.checkpoint_benchmark",
                "--model-size",
                "medium",
                "--context-length",
                "2048",
                "--batch-size",
                "1",
                "--block-size",
                str(block),
                "--output",
                str(output),
            ]
            if run(command, output) not in (0, 1):
                failures.append(str(output))

    if not args.skip_compile:
        for sequence, dim in ((512, 64), (2048, 128), (8192, 128)):
            for phase in ("forward", "backward", "forward_backward"):
                for implementation in ("eager", "compiled"):
                    output = results / "compile" / (
                        f"attention_{implementation}_{sequence}_{dim}_{phase}.json"
                    )
                    command = [
                        python,
                        "-m",
                        "student_scripts.a2k.compile_benchmark",
                        "--kind",
                        "attention",
                        "--implementation",
                        implementation,
                        "--sequence-length",
                        str(sequence),
                        "--head-dim",
                        str(dim),
                        "--phase",
                        phase,
                        "--output",
                        str(output),
                    ]
                    if run(command, output) not in (0, 1):
                        failures.append(str(output))
        for phase in ("forward", "forward_backward", "train_step"):
            for implementation in ("eager", "compiled"):
                output = results / "compile" / (
                    f"small_model_{implementation}_ctx512_{phase}.json"
                )
                command = [
                    python,
                    "-m",
                    "student_scripts.a2k.compile_benchmark",
                    "--kind",
                    "model",
                    "--implementation",
                    implementation,
                    "--sequence-length",
                    "512",
                    "--phase",
                    phase,
                    "--output",
                    str(output),
                ]
                if run(command, output) not in (0, 1):
                    failures.append(str(output))

    if not args.skip_baseline:
        for sequence in (512, 2048, 8192):
            for dim in (64, 128):
                for phase in ("forward", "backward", "forward_backward"):
                    output = results / "attention_baseline" / (
                        f"eager_{sequence}_{dim}_{phase}.json"
                    )
                    command = [
                        python,
                        "-m",
                        "student_scripts.a2k.attention_benchmark",
                        "--sequence-length",
                        str(sequence),
                        "--head-dim",
                        str(dim),
                        "--phase",
                        phase,
                        "--output",
                        str(output),
                    ]
                    if run(command, output) not in (0, 1):
                        failures.append(str(output))

    if not args.skip_flash:
        for sequence in (512, 2048, 8192, 16384):
            for dim in (64, 128):
                implementations = ("eager", "compiled", "triton") if sequence < 16384 else ("eager", "triton")
                for implementation in implementations:
                    for phase in ("forward", "backward", "forward_backward"):
                        output = results / "flash" / f"{implementation}_{sequence}_{dim}_{phase}.json"
                        command = [
                            python,
                            "-m",
                            "student_scripts.a2k.flash_benchmark",
                            "--implementation",
                            implementation,
                            "--sequence-length",
                            str(sequence),
                            "--head-dim",
                            str(dim),
                            "--phase",
                            phase,
                            "--output",
                            str(output),
                        ]
                        if run(command, output) not in (0, 1):
                            failures.append(str(output))

    if not args.skip_correctness:
        correctness = results / "correctness.json"
        command = [
            python,
            "-m",
            "student_scripts.a2k.correctness",
            "--output",
            str(correctness),
        ]
        if run(command, correctness) not in (0, 1):
            failures.append(str(correctness))

    evidence = results / "memory_evidence.json"
    run(
        [
            python,
            "-m",
            "student_scripts.a2k.memory_evidence",
            "--results",
            str(results),
            "--output",
            str(evidence),
        ],
        evidence,
    )
    print(json.dumps({"failures": failures, "results": str(results)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
