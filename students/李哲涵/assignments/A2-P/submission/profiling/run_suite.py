from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _run(
    module: str,
    arguments: list[str],
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", module, *arguments]
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if output_path.is_file():
        try:
            result = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            result = {"status": "failed", "error": "result JSON is invalid"}
    else:
        result = {"status": "failed", "error": "result JSON was not created"}
    result["launcher_returncode"] = completed.returncode
    print(
        f"DONE {output_path.name}: {result.get('status', 'unknown')}",
        flush=True,
    )
    return result


def run_e2e(root: Path, device: str) -> None:
    common = [
        "--model-size",
        "small",
        "--batch-size",
        "4",
        "--context-length",
        "512",
        "--dtype",
        "fp32",
        "--device",
        device,
        "--steps",
        "10",
    ]
    for mode in ("forward", "forward_backward", "train_step"):
        name = f"small_{mode}_warmup5.json"
        _run(
            "profiling.benchmark",
            [*common, "--mode", mode, "--warmup", "5", "--output", str(root / name)],
            root / name,
            root / "logs" / name.replace(".json", ".log"),
        )
    name = "small_train_step_warmup0.json"
    _run(
        "profiling.benchmark",
        [*common, "--mode", "train_step", "--warmup", "0", "--output", str(root / name)],
        root / name,
        root / "logs" / name.replace(".json", ".log"),
    )


def run_mixed(root: Path, device: str) -> None:
    output = root / "mixed_precision.json"
    _run(
        "profiling.mixed_precision",
        [
            "--device",
            device,
            "--output",
            str(output),
            "--warmup",
            "5",
            "--steps",
            "10",
            "--seed",
            "0",
            "--model-size",
            "small",
            "--batch-size",
            "4",
            "--context-length",
            "512",
        ],
        output,
        root / "logs" / "mixed_precision.log",
    )


def run_profiles(root: Path, device: str) -> None:
    profile_root = root / "profile"
    for model_size in ("small", "medium"):
        for context_length in (256, 512, 1024):
            name = f"{model_size}_train_step_s{context_length}"
            trace = profile_root / "traces" / f"{name}.json"
            summary = profile_root / "summaries" / f"{name}.json"
            _run(
                "profiling.compute_profile",
                [
                    "--model-size",
                    model_size,
                    "--batch-size",
                    "1",
                    "--context-length",
                    str(context_length),
                    "--warmup",
                    "5",
                    "--dtype",
                    "fp32",
                    "--device",
                    device,
                    "--trace",
                    str(trace),
                    "--summary",
                    str(summary),
                ],
                summary,
                profile_root / "logs" / f"{name}.log",
            )


def _memory_case(
    root: Path,
    device: str,
    model_size: str,
    mode: str,
    dtype: str,
    context_length: int,
    label: str | None = None,
) -> dict[str, Any]:
    name = label or f"{model_size}_{mode}_{dtype}_s{context_length}"
    output = root / "memory" / "runs" / f"{name}.json"
    snapshot = root / "memory" / "snapshots" / f"{name}.pickle"
    return _run(
        "profiling.memory_snapshot",
        [
            "--model-size",
            model_size,
            "--mode",
            mode,
            "--dtype",
            dtype,
            "--batch-size",
            "1",
            "--context-length",
            str(context_length),
            "--warmup",
            "1",
            "--warmup-mode",
            "forward",
            "--device",
            device,
            "--seed",
            "0",
            "--snapshot",
            str(snapshot),
            "--output",
            str(output),
        ],
        output,
        root / "memory" / "logs" / f"{name}.log",
    )


def run_memory(root: Path, device: str) -> None:
    for dtype in ("fp32", "bf16"):
        for context_length in (128, 2048):
            for mode in ("forward", "train_step"):
                result = _memory_case(root, device, "xl", mode, dtype, context_length)
                if context_length != 2048 or result.get("status") not in {"oom", "failed"}:
                    continue
                # The assignment specifies this fallback order. Keep each
                # fallback under a distinct label instead of relabeling it.
                _memory_case(
                    root,
                    device,
                    "xl",
                    mode,
                    dtype,
                    1024,
                    label=f"fallback_xl_{mode}_{dtype}_s1024",
                )
                _memory_case(
                    root,
                    device,
                    "large",
                    mode,
                    dtype,
                    2048,
                    label=f"fallback_large_{mode}_{dtype}_s2048",
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the A2-P experiment matrix")
    parser.add_argument(
        "--suite",
        choices=("e2e", "mixed", "profile", "memory", "all"),
        default="all",
    )
    parser.add_argument("--output-root", default="results/a2p")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = (ROOT / args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.suite in {"e2e", "all"}:
        run_e2e(root / "benchmark", args.device)
    if args.suite in {"mixed", "all"}:
        run_mixed(root, args.device)
    if args.suite in {"profile", "all"}:
        run_profiles(root, args.device)
    if args.suite in {"memory", "all"}:
        run_memory(root, args.device)
    print(f"SUITE COMPLETE: {args.suite}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

