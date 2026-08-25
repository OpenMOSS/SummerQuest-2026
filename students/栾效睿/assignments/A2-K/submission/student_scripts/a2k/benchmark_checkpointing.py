from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from student_scripts.a2k.utils import ALLOCATOR_LIMIT_BYTES, MIB, allocator_evidence, benchmark_cuda_step, configure_cuda, cuda_peak_mib, is_cuda_oom, load_json, refresh_memory_summary, seed_all, write_csv, write_json


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "checkpointing.csv"
METADATA_PATH = RESULTS_DIR / "run_metadata.json"
MEMORY_PATH = RESULTS_DIR / "memory_evidence.json"

SEED = 336
WARMUP_STEPS = 3
MEASUREMENT_STEPS = 5
MODEL_CONFIG = {
    "vocab_size": 10_000,
    "d_model": 1_024,
    "d_ff": 4_096,
    "num_layers": 24,
    "num_heads": 16,
}
_V, _D, _F, _N = (MODEL_CONFIG[key] for key in ("vocab_size", "d_model", "d_ff", "num_layers"))
NUM_PARAMETERS = 2 * _V * _D + _N * (4 * _D**2 + 3 * _D * _F + 2 * _D) + _D
BATCH_SIZE = 1
STANDARD_CONTEXT = 1_024
BOUNDARY_CONTEXT = 2_048
BLOCK_SIZES = (1, 2, 4, 8)

CSV_FIELDS = (
    "config_id",
    "model_size",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "checkpoint_block_size",
    "nested",
    "warmup_steps",
    "measurement_steps",
    "step_time_ms_samples",
    "step_time_ms_p50",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark activation checkpointing for A2-K task 1.")
    parser.add_argument("--_context", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_block-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def row_template(context: int, block_size: int | None) -> dict[str, Any]:
    return {
        "config_id": f"medium_ctx{context}_block{block_size if block_size is not None else 'none'}",
        "model_size": "medium",
        "num_layers": MODEL_CONFIG["num_layers"],
        "context_length": context,
        "batch_size": BATCH_SIZE,
        "dtype": "bf16_autocast_fp32_params",
        "checkpoint_block_size": block_size,
        "nested": False,
        "warmup_steps": WARMUP_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "step_time_ms_samples": "[]",
        "step_time_ms_p50": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "not_started",
    }


def model_forward(model: BasicsTransformerLM, tokens: torch.Tensor, block_size: int | None) -> torch.Tensor:
    hidden = model.token_embeddings(tokens)
    if block_size is None:
        for layer in model.layers:
            hidden = layer(hidden)
    else:
        for start in range(0, len(model.layers), block_size):
            stop = min(start + block_size, len(model.layers))
            layers = tuple(model.layers[index] for index in range(start, stop))

            def run_group(x: torch.Tensor, layers: tuple[torch.nn.Module, ...] = layers) -> torch.Tensor:
                for layer in layers:
                    x = layer(x)
                return x

            hidden = checkpoint(run_group, hidden, use_reentrant=False)

    return model.lm_head(model.ln_final(hidden))


def run_one(context: int, block_size: int | None, output: Path) -> None:
    row = row_template(context, block_size)
    result: dict[str, Any] = {"row": row, "metadata": {}}

    try:
        result["metadata"] = configure_cuda()
        seed_all(SEED)

        model = BasicsTransformerLM(context_length=context, **MODEL_CONFIG).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        tokens = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, context), device="cuda")
        targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, context), device="cuda")

        def train_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = cross_entropy(model_forward(model, tokens, block_size), targets)
            loss.backward()
            optimizer.step()

        times, p50, peak_allocated, peak_reserved = benchmark_cuda_step(
            train_step, WARMUP_STEPS, MEASUREMENT_STEPS
        )
        row.update(
            step_time_ms_samples=json.dumps(times),
            step_time_ms_p50=p50,
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
            status="success" if peak_reserved <= ALLOCATOR_LIMIT_BYTES / MIB else "invalid_allocator_limit",
        )
    except Exception as error:
        row.update(
            status="oom" if is_cuda_oom(error) else f"error:{type(error).__name__}",
            peak_allocated_mib=cuda_peak_mib(),
            peak_reserved_mib=cuda_peak_mib(reserved=True),
        )
        result["error"] = str(error).splitlines()[0][:300]

    write_json(output, result)


def launch(context: int, block_size: int | None, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.benchmark_checkpointing",
        "--_context",
        str(context),
        "--_block-size",
        str(block_size or 0),
        "--_result",
        str(output),
    ]
    label = row_template(context, block_size)["config_id"]
    print(f"running {label}", flush=True)
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if not output.exists():
        return {
            "row": {
                **row_template(context, block_size),
                "status": f"error:child_exit_{process.returncode}",
            },
            "metadata": {},
        }
    result = load_json(output)
    print(f"  {result['row']['status']}", flush=True)
    if str(result["row"]["status"]).startswith("error:"):
        print(f"  {result.get('error', process.stderr.strip())}", file=sys.stderr)
    return result


def write_outputs(results: list[dict[str, Any]]) -> None:
    rows = [result["row"] for result in results]
    write_csv(CSV_PATH, rows, CSV_FIELDS)

    first_metadata = next((result["metadata"] for result in results if result["metadata"]), {})
    metadata = load_json(METADATA_PATH)
    metadata.update(
        {
            "seed": SEED,
            "command": "uv run python -m student_scripts.a2k.benchmark_checkpointing",
            **first_metadata,
        }
    )
    metadata["checkpointing"] = {
        "model_size": "medium",
        "model_config": MODEL_CONFIG,
        "num_parameters": NUM_PARAMETERS,
        "batch_size": BATCH_SIZE,
        "optimizer": "torch.optim.AdamW(lr=1e-3, weight_decay=0.01)",
        "autocast_dtype": "bfloat16",
        "parameter_dtype": "float32",
        "warmup_steps": WARMUP_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "timer": "torch.cuda.Event",
        "torch_compile": False,
        "per_process_start_free_memory_mib": {
            row["config_id"]: result.get("metadata", {}).get("gpu", {}).get("start_free_memory_mib")
            for row, result in zip(rows, results)
        },
    }
    write_json(METADATA_PATH, metadata)

    allocated = [float(row["peak_allocated_mib"]) for row in rows if row["peak_allocated_mib"] != ""]
    reserved = [float(row["peak_reserved_mib"]) for row in rows if row["peak_reserved_mib"] != ""]
    evidence = load_json(MEMORY_PATH)
    evidence["allocator"] = allocator_evidence(metadata.get("allocator", {}).get("fraction"))
    evidence["checkpointing"] = {
        "highest_peak_allocated_mib": max(allocated, default=None),
        "highest_peak_reserved_mib": max(reserved, default=None),
        "within_23gib_allocator": bool(reserved) and max(reserved) <= ALLOCATOR_LIMIT_BYTES / MIB,
        "within_24gib": bool(reserved) and max(reserved) < 24 * 1024,
        "config_status": {row["config_id"]: row["status"] for row in rows},
    }
    refresh_memory_summary(evidence)
    write_json(MEMORY_PATH, evidence)


def run_matrix() -> int:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="a2k_checkpointing_") as temp_dir:
        temp = Path(temp_dir)
        for index, block_size in enumerate((None, *BLOCK_SIZES)):
            results.append(launch(STANDARD_CONTEXT, block_size, temp / f"standard_{index}.json"))

        successful = [
            result
            for result in results
            if result["row"]["checkpoint_block_size"] is not None and result["row"]["status"] == "success"
        ]
        best_block = (
            int(min(successful, key=lambda result: float(result["row"]["peak_allocated_mib"]))["row"]["checkpoint_block_size"])
            if successful
            else None
        )

        results.append(launch(BOUNDARY_CONTEXT, None, temp / "boundary_baseline.json"))
        if best_block is not None:
            results.append(launch(BOUNDARY_CONTEXT, best_block, temp / "boundary_checkpoint.json"))
        else:
            row = row_template(BOUNDARY_CONTEXT, None)
            row.update(
                config_id="medium_ctx2048_selected_checkpoint",
                status="not_run_no_successful_1024_checkpoint",
            )
            results.append({"row": row, "metadata": {}})

    write_outputs(results)
    rows = [result["row"] for result in results]
    boundary_checkpoint_ok = any(
        row["context_length"] == BOUNDARY_CONTEXT
        and row["checkpoint_block_size"] is not None
        and row["status"] == "success"
        for row in rows
    )
    unexpected = any(row["status"] not in {"success", "oom"} for row in rows)
    for path in (CSV_PATH, METADATA_PATH, MEMORY_PATH):
        print(f"wrote {path}")
    return int(unexpected or not boundary_checkpoint_ok)


def main() -> int:
    args = parse_args()
    if args._context is not None:
        if args._result is None:
            raise ValueError("Missing internal result path")
        run_one(args._context, args._block_size or None, args._result)
        return 0
    return run_matrix()


if __name__ == "__main__":
    raise SystemExit(main())
