#!/usr/bin/env python3
"""Run the required A1 baselines, sweeps, and ablations across visible GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def configure_runtime_environment() -> None:
    """Keep all job-local state on the shared project filesystem."""

    runtime_root = ROOT / ".runtime"
    locations = {
        "HOME": runtime_root / "home",
        "XDG_CACHE_HOME": runtime_root / "cache",
        "TMPDIR": runtime_root / "tmp",
        "TORCHINDUCTOR_CACHE_DIR": runtime_root / "torchinductor",
        "TRITON_CACHE_DIR": runtime_root / "triton",
        "UV_CACHE_DIR": runtime_root / "uv-cache",
    }
    for variable, directory in locations.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, default=Path("configs/experiment_suite.json"))
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU IDs; defaults to all visible GPUs.")
    parser.add_argument("--expected-gpus", type=int, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(value, ensure_ascii=False)}"


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a dotted configuration key in an in-memory effective config."""

    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def effective_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return the exact config that ``train_lm.py`` will resolve."""

    result = deepcopy(base)
    for key, value in overrides.items():
        set_dotted(result, key, value)
    return result


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def batch_sweep_is_complete(path: Path, required_batch_sizes: list[int], require_oom: bool) -> bool:
    records = read_jsonl_objects(path)
    by_size = {int(record["batch_size"]): record for record in records if "batch_size" in record}
    required_complete = all(by_size.get(size, {}).get("status") == "completed" for size in required_batch_sizes)
    oom_complete = not require_oom or any(record.get("status") == "oom" for record in records)
    return required_complete and oom_complete


def isolate_task_caches(environment: dict[str, str], task_name: str) -> None:
    """Give concurrently compiled runs separate project-local caches."""

    task_runtime = ROOT / ".runtime" / "experiment_tasks" / task_name
    locations = {
        "TMPDIR": task_runtime / "tmp",
        "TMP": task_runtime / "tmp",
        "TEMP": task_runtime / "tmp",
        "PYTHONPYCACHEPREFIX": task_runtime / "pycache",
        "TORCHINDUCTOR_CACHE_DIR": task_runtime / "torchinductor",
        "TRITON_CACHE_DIR": task_runtime / "triton",
        "CUDA_CACHE_PATH": task_runtime / "cuda",
        "TORCH_EXTENSIONS_DIR": task_runtime / "torch-extensions",
    }
    for variable, directory in locations.items():
        directory.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(directory)


def main() -> int:
    args = parse_args()
    configure_runtime_environment()
    with resolve(args.suite_config).open(encoding="utf-8") as file:
        suite = json.load(file)

    if args.gpus:
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            try:
                count = int(
                    subprocess.check_output(
                        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                        cwd=ROOT,
                        text=True,
                    ).strip()
                )
            except (subprocess.CalledProcessError, ValueError):
                count = 0
            gpu_ids = [str(index) for index in range(count)]
    if not gpu_ids:
        raise RuntimeError("no GPUs were selected or detected")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError("selected GPU identifiers must be unique")
    if args.expected_gpus is not None and len(gpu_ids) != args.expected_gpus:
        raise RuntimeError(f"expected {args.expected_gpus} visible GPUs, found {len(gpu_ids)}")

    baseline = str(suite["baseline_config"])
    with resolve(Path(baseline)).open(encoding="utf-8") as file:
        baseline_definition = json.load(file)
    with resolve(Path(suite["owt_config"])).open(encoding="utf-8") as file:
        owt_definition = json.load(file)
    validation_config = suite.get("validation", {})
    required_batch_sizes = [int(value) for value in validation_config.get("required_batch_sizes", [64, 128])]
    require_batch_oom = bool(validation_config.get("require_batch_oom", True))
    required_inputs = [
        Path(baseline_definition["data"]["train_path"]),
        Path(baseline_definition["data"]["val_path"]),
        Path(owt_definition["data"]["train_path"]),
        Path(owt_definition["data"]["val_path"]),
        Path("tokenizer_artifacts/tinystories/vocab.json"),
        Path("tokenizer_artifacts/tinystories/merges.txt"),
        Path("tokenizer_artifacts/owt/vocab.json"),
        Path("tokenizer_artifacts/owt/merges.txt"),
    ]
    missing_inputs = [
        str(path) for path in required_inputs if not resolve(path).is_file() or resolve(path).stat().st_size == 0
    ]
    if missing_inputs:
        raise FileNotFoundError(f"missing or empty experiment inputs: {', '.join(missing_inputs)}")
    baseline_steps = int(baseline_definition["training"]["max_steps"])
    owt_steps = int(owt_definition["training"]["max_steps"])
    tasks: list[tuple[str, list[str], Path | None, int | None, dict[str, Any] | None]] = [
        (
            "tinystories_baseline",
            [
                sys.executable,
                "scripts/train_lm.py",
                "--config",
                baseline,
                "--device",
                "cuda",
                "--overwrite",
            ],
            Path(baseline_definition["output_dir"]),
            baseline_steps,
            deepcopy(baseline_definition),
        ),
        (
            "owt_baseline",
            [
                sys.executable,
                "scripts/train_lm.py",
                "--config",
                str(suite["owt_config"]),
                "--device",
                "cuda",
                "--overwrite",
            ],
            Path(owt_definition["output_dir"]),
            owt_steps,
            deepcopy(owt_definition),
        ),
    ]
    for ablation in suite["ablations"]:
        name = f"ablation_{ablation['name']}"
        ablation_overrides = {
            "run_name": name,
            "output_dir": f"runs/{name}",
            **ablation["overrides"],
        }
        command = [
            sys.executable,
            "scripts/train_lm.py",
            "--config",
            baseline,
            "--device",
            "cuda",
            "--overwrite",
            "--set",
            override("run_name", name),
            "--set",
            override("output_dir", f"runs/{name}"),
        ]
        for key, value in ablation["overrides"].items():
            command.extend(["--set", override(key, value)])
        tasks.append(
            (
                name,
                command,
                Path(f"runs/{name}"),
                baseline_steps,
                effective_config(baseline_definition, ablation_overrides),
            )
        )

    lr_sweep = suite["learning_rate_sweep"]
    for learning_rate in lr_sweep["values"]:
        label = format(float(learning_rate), ".0e").replace("+", "")
        name = f"lr_{label}"
        lr_overrides = {
            "run_name": name,
            "output_dir": f"runs/lr_sweep/{name}",
            "training.max_lr": learning_rate,
            "training.min_lr": float(learning_rate) * 0.1,
            "training.max_steps": lr_sweep["max_steps"],
            "training.warmup_steps": min(100, int(lr_sweep["max_steps"]) // 10),
            "training.val_every": lr_sweep["val_every"],
            "training.checkpoint_every": 0,
        }
        command = [
            sys.executable,
            "scripts/train_lm.py",
            "--config",
            baseline,
            "--device",
            "cuda",
            "--overwrite",
            "--set",
            override("run_name", name),
            "--set",
            override("output_dir", f"runs/lr_sweep/{name}"),
            "--set",
            override("training.max_lr", learning_rate),
            "--set",
            override("training.min_lr", float(learning_rate) * 0.1),
            "--set",
            override("training.max_steps", lr_sweep["max_steps"]),
            "--set",
            override("training.warmup_steps", min(100, int(lr_sweep["max_steps"]) // 10)),
            "--set",
            override("training.val_every", lr_sweep["val_every"]),
            "--set",
            override("training.checkpoint_every", 0),
        ]
        tasks.append(
            (
                name,
                command,
                Path(f"runs/lr_sweep/{name}"),
                int(lr_sweep["max_steps"]),
                effective_config(baseline_definition, lr_overrides),
            )
        )

    tasks.append(
        (
            "batch_size_sweep",
            [
                sys.executable,
                "scripts/benchmark_batch_sizes.py",
                "--config",
                baseline,
                "--suite-config",
                str(args.suite_config),
                "--device",
                "cuda",
            ],
            None,
            None,
            None,
        )
    )

    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    orchestration_dir = ROOT / "runs" / "orchestration"
    orchestration_dir.mkdir(parents=True, exist_ok=True)

    def run_task(task: tuple[str, list[str], Path | None, int | None, dict[str, Any] | None]) -> dict[str, Any]:
        name, original_command, output_dir, expected_steps, expected_config = task
        command = list(original_command)
        if name == "batch_size_sweep" and batch_sweep_is_complete(
            ROOT / "runs" / "batch_size" / "summary.jsonl",
            required_batch_sizes,
            require_batch_oom,
        ):
            return {
                "name": name,
                "returncode": 0,
                "wall_clock_sec": 0.0,
                "skipped_completed": True,
            }
        if output_dir is not None:
            resolved_output = resolve(output_dir)
            summary_path = resolved_output / "summary.json"
            previous = read_json_object(summary_path)
            previous_config = read_json_object(resolved_output / "resolved_config.json")
            config_matches = expected_config is not None and previous_config == expected_config
            if previous:
                status = previous.get("status")
                completed_steps = int(previous.get("completed_steps", -1))
                divergence_is_result = name.startswith(("lr_", "ablation_"))
                result_is_complete = (status == "diverged" and divergence_is_result) or (
                    status == "completed" and expected_steps is not None and completed_steps >= expected_steps
                )
                if config_matches and result_is_complete:
                    return {
                        "name": name,
                        "returncode": 0,
                        "wall_clock_sec": 0.0,
                        "skipped_completed": True,
                    }
            latest_checkpoint = resolved_output / "latest.pt"
            if config_matches and latest_checkpoint.is_file() and previous.get("status") != "diverged":
                command = [argument for argument in command if argument != "--overwrite"]
                command.extend(["--resume", str(latest_checkpoint)])

        gpu_id = gpu_queue.get()
        started = time.time()
        print(json.dumps({"event": "task_started", "name": name}), flush=True)
        try:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            isolate_task_caches(environment, name)
            log_path = orchestration_dir / f"{name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import sys, torch; "
                            "ok=torch.cuda.is_available() and torch.cuda.device_count()==1; "
                            "print('cuda_worker_ready' if ok else 'cuda_worker_unavailable'); "
                            "sys.exit(0 if ok else 3)"
                        ),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
                if probe.returncode != 0:
                    return {
                        "name": name,
                        "gpu": gpu_id,
                        "returncode": probe.returncode,
                        "wall_clock_sec": time.time() - started,
                        "log": str(log_path.relative_to(ROOT)),
                    }
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while process.poll() is None:
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        print(
                            json.dumps(
                                {
                                    "event": "task_heartbeat",
                                    "name": name,
                                    "wall_clock_sec": time.time() - started,
                                }
                            ),
                            flush=True,
                        )
            return {
                "name": name,
                "gpu": gpu_id,
                "returncode": process.returncode,
                "wall_clock_sec": time.time() - started,
                "log": str(log_path.relative_to(ROOT)),
            }
        finally:
            gpu_queue.put(gpu_id)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    results.sort(key=lambda item: item["name"])
    summary_path = orchestration_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [result for result in results if result["returncode"] != 0]
    validation_errors: list[str] = []

    def read_run_summary(path: Path) -> dict[str, Any]:
        return read_json_object(resolve(path) / "summary.json")

    baseline_validation_loss: float | None = None
    for label, output_dir, expected_steps in (
        ("tinystories_baseline", Path(baseline_definition["output_dir"]), baseline_steps),
        ("owt_baseline", Path(owt_definition["output_dir"]), owt_steps),
    ):
        run_summary = read_run_summary(output_dir)
        if run_summary.get("status") != "completed" or int(run_summary.get("completed_steps", -1)) < expected_steps:
            validation_errors.append(f"{label} did not complete the required steps")
        if label == "tinystories_baseline":
            try:
                baseline_validation_loss = float(run_summary["final_val_loss"])
            except (KeyError, TypeError, ValueError):
                baseline_validation_loss = None
            maximum_loss = float(validation_config.get("tinystories_max_final_val_loss", 1.45))
            if baseline_validation_loss is None or not math.isfinite(baseline_validation_loss):
                validation_errors.append("tinystories_baseline has no finite final validation loss")
            elif baseline_validation_loss > maximum_loss:
                validation_errors.append(
                    f"tinystories_baseline final validation loss {baseline_validation_loss:.6f} "
                    f"exceeds {maximum_loss:.6f}"
                )

    lr_statuses: list[str] = []
    for learning_rate in lr_sweep["values"]:
        label = format(float(learning_rate), ".0e").replace("+", "")
        run_summary = read_run_summary(Path(f"runs/lr_sweep/lr_{label}"))
        status = str(run_summary.get("status", "missing"))
        lr_statuses.append(status)
        if status not in {"completed", "diverged"}:
            validation_errors.append(f"learning-rate run {label} has no valid summary")
    divergent_lr_runs = [
        format(float(rate), ".0e").replace("+", "")
        for rate, status in zip(lr_sweep["values"], lr_statuses, strict=True)
        if status == "diverged"
    ]
    if bool(validation_config.get("require_divergent_learning_rate", True)) and not divergent_lr_runs:
        validation_errors.append("learning-rate sweep did not contain a divergent run")

    for ablation in suite["ablations"]:
        name = f"ablation_{ablation['name']}"
        run_summary = read_run_summary(Path(f"runs/{name}"))
        if run_summary.get("status") not in {"completed", "diverged"}:
            validation_errors.append(f"{name} has no valid summary")

    batch_path = ROOT / "runs" / "batch_size" / "summary.jsonl"
    batch_records = read_jsonl_objects(batch_path)
    batch_by_size = {int(record["batch_size"]): record for record in batch_records if "batch_size" in record}
    for required_batch_size in required_batch_sizes:
        if batch_by_size.get(required_batch_size, {}).get("status") != "completed":
            validation_errors.append(f"batch size {required_batch_size} was not completed")
    if require_batch_oom and not any(record.get("status") == "oom" for record in batch_records):
        validation_errors.append("batch-size sweep did not reach an out-of-memory boundary")

    validation = {
        "status": "error" if failed or validation_errors else "ok",
        "process_failures": [result["name"] for result in failed],
        "validation_errors": validation_errors,
        "checks": {
            "tinystories_final_val_loss": baseline_validation_loss,
            "tinystories_max_final_val_loss": float(validation_config.get("tinystories_max_final_val_loss", 1.45)),
            "divergent_learning_rate_runs": divergent_lr_runs,
            "required_batch_sizes": required_batch_sizes,
            "batch_oom_observed": any(record.get("status") == "oom" for record in batch_records),
        },
    }
    (orchestration_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if failed or validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
