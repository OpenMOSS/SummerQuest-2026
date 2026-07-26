from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3, "10b": 4}
MODE_ORDER = {"forward": 0, "forward_backward": 1, "train_step": 2}
DTYPE_ORDER = {"fp32": 0, "bf16": 1, "fp16": 2}
NSYS_PATTERN = re.compile(r"(?P<size>small|large)_ctx(?P<context>\d+)_(?P<mode>forward|forward_backward|train_step)_fp32$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A2 profiling results as Markdown tables.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return lines


def format_phase(metadata: dict[str, Any], phase: str) -> str:
    stats = metadata["statistics"]["phases"].get(phase)
    if stats is None:
        return "—"
    return f"{stats['mean_ms']:.3f} ± {stats['std_ms']:.3f}"


def format_peak_gib(metadata: dict[str, Any]) -> str:
    return f"{metadata['statistics']['peak_memory_allocated_bytes'] / 1024**3:.2f}"


def benchmark_section(results_dir: Path) -> list[str]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in results_dir.glob("timings_*.metadata.json"):
        records.append((path, load_json(path)))

    baseline = [
        (path, metadata)
        for path, metadata in records
        if "_warmup" not in path.name
        and metadata["config"]["context_length"] == 512
        and metadata["config"]["batch_size"] == 4
        and metadata["config"]["warmup"] == 5
        and metadata["config"]["steps"] == 10
    ]
    baseline.sort(
        key=lambda item: (
            SIZE_ORDER[item[1]["config"]["model_size"]],
            DTYPE_ORDER[item[1]["config"]["dtype"]],
            MODE_ORDER[item[1]["config"]["mode"]],
        )
    )

    rows = []
    for _path, metadata in baseline:
        config = metadata["config"]
        rows.append(
            [
                config["model_size"],
                config["dtype"],
                config["mode"],
                format_phase(metadata, "forward"),
                format_phase(metadata, "backward"),
                format_phase(metadata, "optimizer"),
                format_phase(metadata, "total"),
                format_peak_gib(metadata),
            ]
        )

    warmup_records = [metadata for path, metadata in records if path.name == "timings_small_forward_fp32.metadata.json" or "_warmup" in path.name]
    warmup_records.sort(key=lambda metadata: metadata["config"]["warmup"])
    warmup_rows = [
        [
            str(metadata["config"]["warmup"]),
            format_phase(metadata, "forward"),
            f"{metadata['statistics']['phases']['forward']['cv'] * 100:.2f}%",
            f"{metadata['statistics']['phases']['forward']['min_ms']:.3f}",
            f"{metadata['statistics']['phases']['forward']['max_ms']:.3f}",
        ]
        for metadata in warmup_records
    ]

    return [
        "## End-to-end benchmark",
        "",
        "Times are mean ± population standard deviation in milliseconds. Peak is allocated CUDA memory.",
        "",
        *markdown_table(
            ["Model", "Dtype", "Mode", "Forward (ms)", "Backward (ms)", "Optimizer (ms)", "Total (ms)", "Peak (GiB)"],
            rows,
        ),
        "",
        "### Warm-up comparison",
        "",
        *markdown_table(["Warm-up steps", "Forward (ms)", "CV", "Min (ms)", "Max (ms)"], warmup_rows),
        "",
    ]


def mixed_precision_section(results_dir: Path) -> list[str]:
    path = results_dir / "mixed_precision.json"
    if not path.exists():
        return []
    data = load_json(path)
    accumulation_rows = [
        [
            case["name"],
            case["accumulator_dtype"],
            case["input_dtype"],
            f"{case['result']:.10f}",
            f"{case['absolute_error_from_10']:.10f}",
        ]
        for case in data["accumulation"]
    ]
    toy = data["toy_model"]
    toy_rows = [
        ["Parameters inside autocast", toy["parameter_dtype_inside_autocast"]],
        ["fc1 output", toy["fc1_output"]],
        ["LayerNorm output", toy["layer_norm_output"]],
        ["Logits", toy["logits_dtype"]],
        ["Loss", toy["loss_dtype"]],
        ["Gradients", ", ".join(toy["gradient_dtypes"])],
    ]
    return [
        "## Mixed precision details",
        "",
        "### Accumulation",
        "",
        *markdown_table(["Case", "Accumulator", "Input", "Result", "Absolute error"], accumulation_rows),
        "",
        "### ToyModel dtypes",
        "",
        *markdown_table(["Component", "Observed dtype"], toy_rows),
        "",
    ]


def find_nvtx_ms(rows: list[dict[str, str]], range_name: str) -> str:
    for row in rows:
        if row["Range"].lstrip(":") == range_name:
            return f"{int(row['Total Time (ns)']) / 1e6:.3f}"
    return "—"


def shorten_kernel_name(name: str, limit: int = 80) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 1] + "…"


def nsys_section(results_dir: Path) -> list[str]:
    nsys_dir = results_dir / "nsys"
    records = []
    for kernel_path in nsys_dir.glob("*_cuda_gpu_kern_sum.csv"):
        prefix = kernel_path.name.removesuffix("_cuda_gpu_kern_sum.csv")
        match = NSYS_PATTERN.fullmatch(prefix)
        if match is None:
            continue
        nvtx_path = nsys_dir / f"{prefix}_nvtx_sum.csv"
        kernel_rows = read_csv(kernel_path)
        nvtx_rows = read_csv(nvtx_path)
        if not kernel_rows:
            continue
        top_kernel = kernel_rows[0]
        records.append(
            {
                "size": match.group("size"),
                "context": int(match.group("context")),
                "mode": match.group("mode"),
                "measure_ms": find_nvtx_ms(nvtx_rows, "profile/measure"),
                "forward_ms": find_nvtx_ms(nvtx_rows, "forward"),
                "backward_ms": find_nvtx_ms(nvtx_rows, "backward"),
                "optimizer_ms": find_nvtx_ms(nvtx_rows, "optimizer"),
                "scores_ms": find_nvtx_ms(nvtx_rows, "attention/scores"),
                "softmax_ms": find_nvtx_ms(nvtx_rows, "attention/softmax"),
                "value_ms": find_nvtx_ms(nvtx_rows, "attention/value"),
                "kernel": shorten_kernel_name(top_kernel["Name"]),
                "kernel_calls": top_kernel["Instances"],
                "kernel_time_percent": top_kernel["Time (%)"],
            }
        )
    records.sort(key=lambda item: (SIZE_ORDER[item["size"]], item["context"], MODE_ORDER[item["mode"]]))

    rows = [
        [
            record["size"],
            str(record["context"]),
            record["mode"],
            record["measure_ms"],
            record["forward_ms"],
            record["backward_ms"],
            record["optimizer_ms"],
            record["scores_ms"],
            record["softmax_ms"],
            record["value_ms"],
            record["kernel_time_percent"],
            record["kernel_calls"],
            record["kernel"],
        ]
        for record in records
    ]
    return [
        "## Nsight Systems",
        "",
        "NVTX and kernel times are from one captured measurement step.",
        "",
        *markdown_table(
            [
                "Model",
                "Context",
                "Mode",
                "Measure (ms)",
                "Forward (ms)",
                "Backward (ms)",
                "Optimizer (ms)",
                "Attn scores (ms)",
                "Softmax (ms)",
                "Attn value (ms)",
                "Top kernel (%)",
                "Calls",
                "Top kernel",
            ],
            rows,
        ),
        "",
    ]


def oom_section(results_dir: Path) -> list[str]:
    rows = []
    for path in sorted(results_dir.rglob("*.oom.json")):
        data = load_json(path)
        experiment = data["experiment"]
        failure = data["failure"]
        rows.append(
            [
                experiment["model_size"],
                str(experiment["context_length"]),
                experiment["mode"],
                experiment["dtype"],
                failure["stage"],
                failure["operation"],
                path.name,
            ]
        )
    if not rows:
        return []
    return [
        "## Observed OOM boundaries",
        "",
        *markdown_table(["Model", "Context", "Mode", "Dtype", "Stage", "Operation", "Evidence"], rows),
        "",
    ]


def memory_section(results_dir: Path) -> list[str]:
    records = []
    for path in (results_dir / "memory").glob("*.metadata.json"):
        data = load_json(path)
        config = data["config"]
        after = data["memory_after_measurement"]
        records.append(
            [
                str(config["context_length"]),
                config["mode"],
                config["dtype"],
                data["status"],
                f"{after['peak_allocated_bytes'] / 1024**3:.2f}",
                f"{after['peak_reserved_bytes'] / 1024**3:.2f}",
                "yes" if data["snapshot"]["saved"] else "no",
            ]
        )
    records.sort(key=lambda row: (int(row[0]), MODE_ORDER[row[1]], DTYPE_ORDER[row[2]]))
    if not records:
        return []
    return [
        "## Memory snapshots",
        "",
        *markdown_table(["Context", "Mode", "Dtype", "Status", "Peak allocated (GiB)", "Peak reserved (GiB)", "Snapshot"], records),
        "",
    ]


def nsys_memory_section(results_dir: Path) -> list[str]:
    database_path = results_dir / "nsys" / "memory_xl_ctx128_train_step_fp32.sqlite"
    if not database_path.exists():
        return []

    connection = sqlite3.connect(database_path)
    try:
        phase_rows = []
        for phase in ("forward", "backward", "optimizer"):
            bounds = connection.execute(
                "SELECT start, end FROM NVTX_EVENTS WHERE text = ? ORDER BY (end - start) DESC LIMIT 1",
                (phase,),
            ).fetchone()
            if bounds is None:
                continue
            count, total_bytes, largest_bytes = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(bytes), 0)
                FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                WHERE start >= ? AND start <= ?
                """,
                bounds,
            ).fetchone()
            phase_rows.append(
                [
                    phase,
                    str(count),
                    f"{total_bytes / 1024**3:.2f}",
                    f"{largest_bytes / 1024**2:.1f}",
                ]
            )

        block_rows = []
        for index in range(32):
            range_name = f"transformer_block/{index:02d}/forward"
            bounds = connection.execute(
                "SELECT start, end FROM NVTX_EVENTS WHERE text = ? ORDER BY (end - start) DESC LIMIT 1",
                (range_name,),
            ).fetchone()
            if bounds is None:
                continue
            count, total_bytes, largest_bytes = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(bytes), 0)
                FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                WHERE start >= ? AND start <= ?
                """,
                bounds,
            ).fetchone()
            block_rows.append(
                [
                    str(index),
                    str(count),
                    f"{total_bytes / 1024**2:.1f}",
                    f"{largest_bytes / 1024**2:.1f}",
                ]
            )
    finally:
        connection.close()

    return [
        "## Nsight CUDA memory trace",
        "",
        "Values are CUDA device-memory allocation requests observed inside each NVTX interval. They describe allocator growth, not individual live PyTorch tensors.",
        "",
        *markdown_table(["Phase", "Allocations", "Requested (GiB)", "Largest (MiB)"], phase_rows),
        "",
        "### TransformerBlock forward ranges",
        "",
        *markdown_table(["Block", "Allocations", "Requested (MiB)", "Largest (MiB)"], block_rows),
        "",
    ]


def main() -> None:
    args = parse_args()
    sections = [
        "# Generated profiling summary",
        "",
        *benchmark_section(args.results_dir),
        *mixed_precision_section(args.results_dir),
        *nsys_section(args.results_dir),
        *oom_section(args.results_dir),
        *memory_section(args.results_dir),
        *nsys_memory_section(args.results_dir),
    ]
    text = "\n".join(sections).rstrip() + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"saved summary: {args.output}")


if __name__ == "__main__":
    main()
