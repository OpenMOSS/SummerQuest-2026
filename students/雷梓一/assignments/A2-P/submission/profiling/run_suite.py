from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def python_module(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def memory_status(path: Path) -> str:
    return json.loads(path.read_text())["status"]


def run_memory(root: Path, device: str, model_size: str, context: int, mode: str) -> Path:
    name = f"{model_size}_ctx{context}_{mode}"
    summary = root / "memory" / f"{name}_summary.json"
    run(
        python_module(
            "profiling.memory_snapshot", "--model-size", model_size, "--batch-size", "1",
            "--context-length", str(context), "--mode", mode, "--warmup", "1", "--steps", "1",
            "--dtype", "fp32", "--device", device,
            "--output", str(root / "memory" / f"{name}_unused.json"),
            "--snapshot-output", str(root / "memory" / f"{name}.pickle"),
            "--summary-output", str(summary),
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete A2-P experiment matrix")
    parser.add_argument("--phase", choices=("benchmark", "profile", "mixed", "memory", "all"), default="all")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.results

    if args.phase in ("benchmark", "all"):
        for mode in ("forward", "forward_backward", "train_step"):
            run(
                python_module(
                    "profiling.benchmark", "--model-size", "small", "--batch-size", "4", "--context-length", "512",
                    "--mode", mode, "--warmup", "5", "--steps", "10", "--dtype", "fp32", "--device", args.device,
                    "--output", str(root / "benchmark_raw" / f"small_ctx512_{mode}_w5.json"),
                )
            )
        run(
            python_module(
                "profiling.benchmark", "--model-size", "small", "--batch-size", "4", "--context-length", "512",
                "--mode", "train_step", "--warmup", "0", "--steps", "10", "--dtype", "fp32", "--device", args.device,
                "--output", str(root / "benchmark_raw" / "small_ctx512_train_step_w0.json"),
            )
        )

    if args.phase in ("profile", "all"):
        for model_size in ("small", "medium"):
            for context in (256, 512, 1024):
                name = f"{model_size}_ctx{context}"
                run(
                    python_module(
                        "profiling.profile_runner", "--model-size", model_size, "--batch-size", "1",
                        "--context-length", str(context), "--mode", "train_step", "--warmup", "5", "--steps", "1",
                        "--dtype", "fp32", "--device", args.device,
                        "--output", str(root / "profile" / f"{name}_unused.json"),
                        "--trace-output", str(root / "profile" / f"{name}.json"),
                        "--summary-output", str(root / "profile" / f"{name}_summary.json"),
                    )
                )

    if args.phase in ("mixed", "all"):
        run(
            python_module(
                "profiling.mixed_precision", "--model-size", "small", "--batch-size", "4", "--context-length", "512",
                "--mode", "train_step", "--warmup", "5", "--steps", "10", "--device", args.device,
                "--output", str(root / "mixed" / "benchmark.json"),
                "--combined-output", str(root / "mixed" / "mixed_precision.json"),
            )
        )

    if args.phase in ("memory", "all"):
        for context in (128, 2048):
            for mode in ("forward", "train_step"):
                run_memory(root, args.device, "xl", context, mode)

        required_long = root / "memory" / "xl_ctx2048_train_step_summary.json"
        if memory_status(required_long) != "ok":
            fallback_one = run_memory(root, args.device, "xl", 1024, "train_step")
            if memory_status(fallback_one) != "ok":
                run_memory(root, args.device, "large", 2048, "train_step")


if __name__ == "__main__":
    main()
