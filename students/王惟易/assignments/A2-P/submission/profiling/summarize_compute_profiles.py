import argparse
import csv
import json
from pathlib import Path


PHASE_ORDER = {
    "train_step": 0,
    "zero_grad": 1,
    "forward": 2,
    "loss": 3,
    "backward": 4,
    "optimizer": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate torch.profiler analysis and metadata artifacts.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--profiles-csv", type=Path, required=True)
    parser.add_argument("--attention-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def short_kernel_name(name: str) -> str:
    if len(name) <= 64:
        return name
    if "vectorized_elementwise_kernel" in name and "MulFunctor" in name:
        return "vectorized elementwise MulFunctor"
    if "vectorized_elementwise_kernel" in name and "exp_kernel_cuda" in name:
        return "vectorized exp"
    if "reduce_kernel" in name:
        return "reduction kernel"
    if "elementwise_kernel" in name:
        return "elementwise kernel"
    return name[:61] + "..."


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metadata_paths = sorted(args.input_dir.glob("*.metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no metadata JSON files found in {args.input_dir}")

    profile_rows = []
    attention_rows = []
    environments = []
    marker_sets = []
    model_widths = {}

    for metadata_path in metadata_paths:
        stem = metadata_path.name.removesuffix(".metadata.json")
        analysis_path = args.input_dir / f"{stem}.analysis.json"
        if not analysis_path.exists():
            raise FileNotFoundError(f"analysis JSON does not exist: {analysis_path}")

        metadata = load_json(metadata_path)
        analysis = load_json(analysis_path)
        config = metadata["config"]
        model_widths[config["model_size"]] = config["d_model"]
        environments.append(metadata["environment"])
        marker_sets.append(tuple(analysis["matrix_multiply_kernel_markers"]))

        measurement = analysis["measurement"]
        stages = {
            "train_step": measurement,
            **measurement["phases"],
        }
        for phase, profile in stages.items():
            kernels = profile["kernels"]
            matrix_multiply = kernels["matrix_multiply"]
            other = kernels["other"]
            top_kernel = kernels["top"][0] if kernels["top"] else None
            top_other = other["top"][0] if other["top"] else None
            profile_rows.append(
                {
                    "model_size": config["model_size"],
                    "context_length": config["context_length"],
                    "phase": phase,
                    "cpu_span_ms": profile["cpu_span_ms"],
                    "kernel_span_ms": kernels["span_ms"],
                    "kernel_calls": kernels["calls"],
                    "kernel_cumulative_ms": kernels["cumulative_ms"],
                    "matrix_multiply_calls": matrix_multiply["calls"],
                    "matrix_multiply_cumulative_ms": matrix_multiply["cumulative_ms"],
                    "matrix_multiply_fraction": matrix_multiply["fraction"],
                    "other_calls": other["calls"],
                    "other_cumulative_ms": other["cumulative_ms"],
                    "other_fraction": other["fraction"],
                    "top_kernel": top_kernel["name"] if top_kernel else "",
                    "top_kernel_calls": top_kernel["calls"] if top_kernel else 0,
                    "top_kernel_cumulative_ms": top_kernel["cumulative_ms"] if top_kernel else 0.0,
                    "top_other_kernel": top_other["name"] if top_other else "",
                    "top_other_kernel_calls": top_other["calls"] if top_other else 0,
                    "top_other_kernel_cumulative_ms": top_other["cumulative_ms"] if top_other else 0.0,
                }
            )

        attention = measurement["attention"]
        scores = attention["attention/scores"]
        softmax = attention["attention/softmax"]
        value = attention["attention/value"]
        attention_rows.append(
            {
                "model_size": config["model_size"],
                "context_length": config["context_length"],
                "num_layers": config["num_layers"],
                "scores_calls": scores["gpu_ranges"]["calls"],
                "scores_gpu_span_ms": scores["gpu_ranges"]["total_ms"],
                "scores_kernel_cumulative_ms": scores["kernels"]["cumulative_ms"],
                "softmax_calls": softmax["gpu_ranges"]["calls"],
                "softmax_gpu_span_ms": softmax["gpu_ranges"]["total_ms"],
                "softmax_kernel_cumulative_ms": softmax["kernels"]["cumulative_ms"],
                "value_calls": value["gpu_ranges"]["calls"],
                "value_gpu_span_ms": value["gpu_ranges"]["total_ms"],
                "value_kernel_cumulative_ms": value["kernels"]["cumulative_ms"],
                "softmax_to_scores": softmax["kernels"]["cumulative_ms"] / scores["kernels"]["cumulative_ms"],
                "softmax_to_value": softmax["kernels"]["cumulative_ms"] / value["kernels"]["cumulative_ms"],
            }
        )

    profile_rows.sort(key=lambda row: (row["model_size"], row["context_length"], PHASE_ORDER[row["phase"]]))
    attention_rows.sort(key=lambda row: (row["model_size"], row["context_length"]))

    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("profile environments are inconsistent")
    if any(markers != marker_sets[0] for markers in marker_sets[1:]):
        raise ValueError("matrix-multiply kernel markers are inconsistent")

    write_csv(args.profiles_csv, profile_rows)
    write_csv(args.attention_csv, attention_rows)

    train_step_rows = []
    for row in profile_rows:
        if row["phase"] != "train_step":
            continue
        train_step_rows.append(
            [
                str(row["model_size"]),
                str(row["context_length"]),
                f"{row['cpu_span_ms']:.3f}",
                f"{row['kernel_span_ms']:.3f}",
                str(row["kernel_calls"]),
                f"{row['kernel_cumulative_ms']:.3f}",
                f"{row['matrix_multiply_fraction']:.2%}",
                short_kernel_name(str(row["top_kernel"])),
                str(row["top_kernel_calls"]),
            ]
        )

    attention_markdown_rows = []
    for row in attention_rows:
        attention_markdown_rows.append(
            [
                str(row["model_size"]),
                str(row["context_length"]),
                f"{row['scores_kernel_cumulative_ms']:.3f}",
                f"{row['softmax_kernel_cumulative_ms']:.3f}",
                f"{row['value_kernel_cumulative_ms']:.3f}",
                f"{row['softmax_to_scores']:.2f}",
                f"{row['softmax_to_value']:.2f}",
            ]
        )

    representative_key = max(
        ((row["model_size"], row["context_length"]) for row in profile_rows),
        key=lambda key: (model_widths[key[0]], key[1]),
    )
    phase_rows = []
    for row in profile_rows:
        if (row["model_size"], row["context_length"]) != representative_key:
            continue
        phase_rows.append(
            [
                str(row["phase"]),
                f"{row['cpu_span_ms']:.3f}",
                f"{row['kernel_span_ms']:.3f}",
                str(row["kernel_calls"]),
                f"{row['kernel_cumulative_ms']:.3f}",
                f"{row['matrix_multiply_fraction']:.2%}",
                short_kernel_name(str(row["top_kernel"])),
            ]
        )

    environment = environments[0]
    markdown = "\n\n".join(
        [
            "# Compute profile summary",
            f"Environment: {environment['gpu']}; PyTorch {environment['pytorch']}; CUDA {environment['cuda_runtime']}; matrix-multiply markers: {', '.join(marker_sets[0])}.",
            "## Train-step traces\n\n"
            + markdown_table(
                ["Model", "T", "CPU span ms", "Kernel span ms", "Kernel calls", "Kernel cumulative ms", "Matmul share", "Top kernel", "Calls"],
                train_step_rows,
            ),
            f"## Representative phases ({representative_key[0]}, T={representative_key[1]})\n\n"
            + markdown_table(
                ["Phase", "CPU span ms", "Kernel span ms", "Kernel calls", "Kernel cumulative ms", "Matmul share", "Top kernel"],
                phase_rows,
            ),
            "## Attention subphases\n\n"
            + markdown_table(
                ["Model", "T", "Scores ms", "Softmax ms", "Value ms", "Softmax / scores", "Softmax / value"],
                attention_markdown_rows,
            ),
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown + "\n", encoding="utf-8")

    print(f"profile rows: {len(profile_rows)} -> {args.profiles_csv}")
    print(f"attention rows: {len(attention_rows)} -> {args.attention_csv}")
    print(f"markdown summary: {args.markdown}")


if __name__ == "__main__":
    main()
