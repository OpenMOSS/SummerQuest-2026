"""Run and report controlled TinyStories Transformer architecture ablations."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from scripts.train import TrainConfig, run_training


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens_to_loss(run_dir: Path, target_loss: float) -> int | None:
    for record in _load_records(run_dir / "metrics.jsonl"):
        if record.get("event") == "validation" and record["loss"] <= target_loss:
            return int(record["tokens"])
    return None


def _format_loss(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _smoothed(values: list[float], window: int = 7) -> list[float]:
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "name",
        "label",
        "category",
        "norm_position",
        "position_encoding",
        "ffn_type",
        "d_ff",
        "parameter_count",
        "learning_rate",
        "status",
        "completed_steps",
        "tokens_processed",
        "final_train_loss",
        "final_validation_loss",
        "best_validation_loss",
        "tokens_to_target",
        "elapsed_seconds",
        "mean_tokens_per_second",
        "divergence_reason",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def _plot_one_curve(axis, row: dict[str, Any], *, event: str, loss_ceiling: float) -> None:
    records = _load_records(Path(row["run_dir"]) / "metrics.jsonl")
    selected = [record for record in records if record.get("event") == event]
    if not selected:
        return
    tokens = [record["tokens"] / 1e6 for record in selected]
    losses = [record["loss"] for record in selected]
    if event == "train":
        losses = _smoothed(losses)
        axis.plot(tokens, losses, linewidth=1.5, label=row["label"])
        if row["status"] == "diverged":
            divergence_loss = row.get("final_train_loss")
            if divergence_loss is not None and math.isfinite(divergence_loss):
                axis.scatter(
                    row["tokens_processed"] / 1e6,
                    min(divergence_loss, loss_ceiling),
                    marker="x",
                    s=65,
                    color="black",
                    zorder=5,
                )
    else:
        axis.plot(tokens, losses, marker="o", markersize=3, linewidth=1.5, label=row["label"])


def _format_axis(axis, *, title: str, loss_ceiling: float) -> None:
    axis.set_title(title)
    axis.set_xlabel("Tokens processed (millions)")
    axis.set_ylabel("Per-token cross-entropy")
    axis.set_ylim(bottom=1.0, top=loss_ceiling)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def _plot_layer_norm(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["name"] == "baseline" or row["category"] == "layer_norm"]
    loss_ceiling = max(5.0, 1.05 * max(row["initial_validation_loss"] for row in selected))
    figure, (train_axis, validation_axis) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for row in selected:
        _plot_one_curve(train_axis, row, event="train", loss_ceiling=loss_ceiling)
        _plot_one_curve(validation_axis, row, event="validation", loss_ceiling=loss_ceiling)
    _format_axis(train_axis, title="RMSNorm ablation: training loss", loss_ceiling=loss_ceiling)
    _format_axis(validation_axis, title="RMSNorm ablation: validation loss", loss_ceiling=loss_ceiling)
    figure.savefig(output_dir / "layer_norm_ablation_curves.png", dpi=180)
    figure.savefig(output_dir / "layer_norm_ablation_curves.pdf")
    plt.close(figure)


def _plot_architectures(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = next(row for row in rows if row["name"] == "baseline")
    comparisons = [
        ("post_norm", "Pre-norm vs. post-norm"),
        ("position", "RoPE vs. NoPE"),
        ("ffn", "SwiGLU vs. SiLU"),
    ]
    loss_ceiling = max(5.0, 1.05 * baseline["initial_validation_loss"])
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for row_index, (category, title) in enumerate(comparisons):
        selected = [baseline, *(row for row in rows if row["category"] == category)]
        for row in selected:
            _plot_one_curve(axes[row_index, 0], row, event="train", loss_ceiling=loss_ceiling)
            _plot_one_curve(axes[row_index, 1], row, event="validation", loss_ceiling=loss_ceiling)
        _format_axis(axes[row_index, 0], title=f"{title}: training", loss_ceiling=loss_ceiling)
        _format_axis(axes[row_index, 1], title=f"{title}: validation", loss_ceiling=loss_ceiling)
    figure.savefig(output_dir / "architecture_ablation_curves.png", dpi=180)
    figure.savefig(output_dir / "architecture_ablation_curves.pdf")
    plt.close(figure)


def _result_sentence(row: dict[str, Any]) -> str:
    if row["status"] == "diverged":
        return f"**{row['label']}** diverged: {row['divergence_reason']}."
    return f"**{row['label']}** completed with final validation loss **{row['final_validation_loss']:.4f}**."


def _write_report(rows: list[dict[str, Any]], output_dir: Path, payload: dict[str, Any]) -> None:
    target_loss = float(payload.get("report_target_loss", 2.0))
    baseline = next(row for row in rows if row["name"] == "baseline")
    table_rows = [
        "| {label} | {norm} | {position} | {ffn} | {dff} | {params:,} | {lr:.3g} | {status} | {valid} | {target} |".format(
            label=row["label"],
            norm=row["norm_position"],
            position=row["position_encoding"],
            ffn=row["ffn_type"],
            dff=row["d_ff"],
            params=row["parameter_count"],
            lr=row["learning_rate"],
            status=row["status"],
            valid=_format_loss(row.get("final_validation_loss")),
            target="-" if row.get("tokens_to_target") is None else f"{row['tokens_to_target']:,}",
        )
        for row in rows
    ]

    no_norm_rows = [row for row in rows if row["category"] == "layer_norm"]
    stable_no_norm = [row for row in no_norm_rows if row["status"] == "completed"]
    previous_lr_no_norm = next(row for row in no_norm_rows if row["learning_rate"] == baseline["learning_rate"])
    layer_findings = [_result_sentence(previous_lr_no_norm)]
    if stable_no_norm:
        best_no_norm = min(stable_no_norm, key=lambda row: row["final_validation_loss"])
        layer_findings.append(
            f"The best stable no-RMSNorm trial used learning rate **{best_no_norm['learning_rate']:.3g}** and "
            f"finished at **{best_no_norm['final_validation_loss']:.4f}**, compared with the normalized baseline's "
            f"**{baseline['final_validation_loss']:.4f}**. Removing RMSNorm therefore changes both the stability "
            "range and the attainable loss under the same token budget."
        )

    post_norm = next(row for row in rows if row["category"] == "post_norm")
    nope = next(row for row in rows if row["category"] == "position")
    silu = next(row for row in rows if row["category"] == "ffn")
    parameter_difference = 100 * (silu["parameter_count"] - baseline["parameter_count"]) / baseline["parameter_count"]

    report = "\n".join(
        [
            "# TinyStories architecture ablations",
            "",
            "## Controlled protocol",
            "",
            f"All comparisons use batch size {payload['base_config']['batch_size']}, peak learning rate "
            f"{payload['base_config']['learning_rate']:.3g} unless explicitly labeled otherwise, and "
            f"{payload['base_config']['total_tokens']:,} training tokens. The pre-norm + RoPE + SwiGLU baseline "
            "is reused from the completed batch-size experiment, so no duplicate baseline training is required.",
            "",
            "## Results",
            "",
            f"| Variant | Norm | Position | FFN | d_ff | Parameters | Peak LR | Status | Final validation loss | Tokens to loss <= {target_loss:.2f} |",
            "|:---|:---|:---|:---|---:|---:|---:|:---|---:|---:|",
            *table_rows,
            "",
            "## RMSNorm ablation",
            "",
            *layer_findings,
            "",
            "## Pre-norm versus post-norm",
            "",
            _result_sentence(post_norm),
            f"The pre-norm baseline finished at **{baseline['final_validation_loss']:.4f}** under the same data and "
            "learning-rate schedule, showing the effect of moving the same normalization operations after the "
            "residual branches.",
            "",
            "## RoPE versus NoPE",
            "",
            _result_sentence(nope),
            f"The RoPE baseline finished at **{baseline['final_validation_loss']:.4f}**. The difference isolates "
            "explicit positional information because the causal mask and every trainable parameter are otherwise "
            "unchanged.",
            "",
            "## SwiGLU versus SiLU",
            "",
            _result_sentence(silu),
            f"The SiLU model uses d_ff=2048 and {silu['parameter_count']:,} parameters, a "
            f"**{parameter_difference:+.2f}%** difference from the {baseline['parameter_count']:,}-parameter "
            "SwiGLU baseline. This keeps the comparison approximately parameter matched while removing gating.",
            "",
            "![RMSNorm ablation curves](layer_norm_ablation_curves.png)",
            "",
            "![Architecture ablation curves](architecture_ablation_curves.png)",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _make_config(base: dict[str, Any], output_dir: Path, overrides: dict[str, Any]) -> TrainConfig:
    field_names = {field.name for field in dataclasses.fields(TrainConfig)}
    unknown = sorted((set(base) | set(overrides)) - field_names)
    if unknown:
        raise ValueError(f"Unknown TrainConfig fields: {', '.join(unknown)}")
    config_payload = base | overrides
    for path_field in ("train_data", "valid_data"):
        config_payload[path_field] = Path(config_payload[path_field])
    config_payload["output_dir"] = output_dir
    return TrainConfig(**config_payload)


def _baseline_row(payload: dict[str, Any], target_loss: float) -> dict[str, Any]:
    run_dir = Path(payload["baseline_run_dir"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    base_config = payload["base_config"]
    return summary | {
        "name": "baseline",
        "label": "Baseline (pre-norm + RoPE + SwiGLU)",
        "category": "baseline",
        "norm_position": "pre",
        "position_encoding": "rope",
        "ffn_type": "swiglu",
        "d_ff": 1344,
        "learning_rate": float(base_config["learning_rate"]),
        "tokens_to_target": _tokens_to_loss(run_dir, target_loss),
        "run_dir": str(run_dir.resolve()),
    }


def run_sweep(config_path: Path, *, resume: bool = False, overwrite: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    target_loss = float(payload.get("report_target_loss", 2.0))
    rows = [_baseline_row(payload, target_loss)]
    for variant in payload["variants"]:
        run_dir = output_dir / variant["name"]
        config = _make_config(payload["base_config"], run_dir, variant.get("overrides", {}))
        summary_path = run_dir / "summary.json"
        if resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            config.overwrite = overwrite
            summary = run_training(config)
        rows.append(
            summary
            | {
                "name": variant["name"],
                "label": variant["label"],
                "category": variant["category"],
                "norm_position": config.norm_position,
                "position_encoding": config.position_encoding,
                "ffn_type": config.ffn_type,
                "d_ff": config.d_ff,
                "tokens_to_target": _tokens_to_loss(run_dir, target_loss),
                "run_dir": str(run_dir.resolve()),
            }
        )

    (output_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_csv(rows, output_dir / "sweep_summary.csv")
    _write_report(rows, output_dir, payload)
    _plot_layer_norm(rows, output_dir)
    _plot_architectures(rows, output_dir)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed variant summaries")
    parser.add_argument("--overwrite", action="store_true", help="Replace files in variant directories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args.config, resume=args.resume, overwrite=args.overwrite)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
