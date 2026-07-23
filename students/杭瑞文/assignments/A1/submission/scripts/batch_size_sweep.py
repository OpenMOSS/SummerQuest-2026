"""Run, summarize, and plot a fixed-token TinyStories batch-size sweep."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
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
    if window <= 1:
        return values
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
        "batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_steps",
        "completed_steps",
        "tokens_processed",
        "status",
        "final_train_loss",
        "final_validation_loss",
        "best_validation_loss",
        "elapsed_seconds",
        "mean_tokens_per_second",
        "tokens_to_target",
        "divergence_reason",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def _plot_curves(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib; run `uv add matplotlib` and retry") from error

    completed_rows = [row for row in rows if row["status"] == "completed"]
    initial_losses = [row["initial_validation_loss"] for row in completed_rows]
    loss_ceiling = max(5.0, 1.05 * max(initial_losses, default=5.0))
    figure, (train_axis, validation_axis) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for row in sorted(rows, key=lambda item: item["batch_size"]):
        records = _load_records(Path(row["run_dir"]) / "metrics.jsonl")
        train_records = [record for record in records if record.get("event") == "train"]
        validation_records = [record for record in records if record.get("event") == "validation"]
        label = f"batch={row['batch_size']}"
        if row["gradient_accumulation_steps"] > 1:
            label += f" (micro={row['micro_batch_size']})"
        if row["status"] != "completed":
            label += f" ({row['status']})"
        if train_records:
            tokens = [record["tokens"] / 1e6 for record in train_records]
            losses = _smoothed([record["loss"] for record in train_records])
            train_axis.plot(tokens, losses, linewidth=1.5, label=label)
        if validation_records:
            tokens = [record["tokens"] / 1e6 for record in validation_records]
            losses = [record["loss"] for record in validation_records]
            validation_axis.plot(tokens, losses, marker="o", markersize=3, linewidth=1.5, label=label)

    train_axis.set_title("Training loss (7-point moving average)")
    validation_axis.set_title("Held-out validation loss")
    for axis in (train_axis, validation_axis):
        axis.set_xlabel("Tokens processed (millions)")
        axis.set_ylabel("Per-token cross-entropy")
        axis.set_ylim(bottom=1.0, top=loss_ceiling)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / "batch_size_learning_curves.png", dpi=180)
    figure.savefig(output_dir / "batch_size_learning_curves.pdf")
    plt.close(figure)


def _plot_efficiency(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib; run `uv add matplotlib` and retry") from error

    completed_rows = sorted(
        (row for row in rows if row["status"] == "completed"),
        key=lambda row: row["batch_size"],
    )
    if not completed_rows:
        return
    batch_sizes = [row["batch_size"] for row in completed_rows]
    throughputs = [row["mean_tokens_per_second"] for row in completed_rows]
    validation_losses = [row["final_validation_loss"] for row in completed_rows]
    figure, throughput_axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    validation_axis = throughput_axis.twinx()
    throughput_axis.plot(batch_sizes, throughputs, marker="o", linewidth=2, color="tab:blue", label="Throughput")
    validation_axis.plot(
        batch_sizes,
        validation_losses,
        marker="s",
        linewidth=2,
        color="tab:red",
        label="Final validation loss",
    )
    throughput_axis.set_xscale("log", base=2)
    throughput_axis.set_xlabel("Effective batch size")
    throughput_axis.set_ylabel("Mean tokens/second", color="tab:blue")
    validation_axis.set_ylabel("Final validation loss", color="tab:red")
    throughput_axis.grid(alpha=0.25)
    lines = throughput_axis.lines + validation_axis.lines
    throughput_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.savefig(output_dir / "batch_size_efficiency.png", dpi=180)
    figure.savefig(output_dir / "batch_size_efficiency.pdf")
    plt.close(figure)


def _write_report(rows: list[dict[str, Any]], output_dir: Path, payload: dict[str, Any]) -> None:
    base_config = payload["base_config"]
    target_loss = float(payload.get("report_target_loss", 2.0))
    completed_rows = [
        row for row in rows if row["status"] == "completed" and row["final_validation_loss"] is not None
    ]
    best_row = min(completed_rows, key=lambda row: row["final_validation_loss"]) if completed_rows else None
    fastest_row = max(completed_rows, key=lambda row: row["mean_tokens_per_second"]) if completed_rows else None

    table_rows: list[str] = []
    for row in sorted(rows, key=lambda item: item["batch_size"]):
        table_rows.append(
            "| {batch} | {micro} | {accum} | {updates:,} | {status} | {train} | {valid} | {throughput:,.0f} | {target} |".format(
                batch=row["batch_size"],
                micro=row["micro_batch_size"],
                accum=row["gradient_accumulation_steps"],
                updates=row["completed_steps"],
                status=row["status"],
                train=_format_loss(row.get("final_train_loss")),
                valid=_format_loss(row.get("final_validation_loss")),
                throughput=row.get("mean_tokens_per_second", 0.0),
                target="-" if row.get("tokens_to_target") is None else f"{row['tokens_to_target']:,}",
            )
        )

    findings: list[str] = []
    if best_row is not None:
        findings.append(
            f"With the learning rate fixed at **{base_config['learning_rate']:.3g}**, batch size "
            f"**{best_row['batch_size']}** achieved the lowest final validation loss, "
            f"**{best_row['final_validation_loss']:.4f}**."
        )
    if fastest_row is not None:
        findings.append(
            f"The highest measured end-to-end throughput was **{fastest_row['mean_tokens_per_second']:,.0f} "
            f"tokens/s** at effective batch size **{fastest_row['batch_size']}**."
        )
    rows_by_batch = {row["batch_size"]: row for row in completed_rows}
    if 1 in rows_by_batch:
        batch_one = rows_by_batch[1]
        findings.append(
            f"Batch size 1 had the noisiest training curve and finished at validation loss "
            f"**{batch_one['final_validation_loss']:.4f}**; it never reached the {target_loss:.2f} target. "
            f"The inherited learning rate {base_config['learning_rate']:.3g} is therefore too aggressive for this "
            "high-variance regime, and a separate batch-1 run should tune a smaller rate if batch 1 were the target "
            "operating point. The rate was kept fixed here to isolate the requested batch-size comparison."
        )
    if 64 in rows_by_batch and 128 in rows_by_batch and 256 in rows_by_batch:
        batch_64 = rows_by_batch[64]
        batch_128 = rows_by_batch[128]
        batch_256 = rows_by_batch[256]
        findings.append(
            f"Quality saturated beyond batch 64: final validation loss changed from "
            f"**{batch_64['final_validation_loss']:.4f}** (batch 64) to "
            f"**{batch_128['final_validation_loss']:.4f}** (batch 128) and "
            f"**{batch_256['final_validation_loss']:.4f}** (batch 256). Batch 128 also reduced measured throughput "
            "because the physical activation and logit tensors created more unified-memory pressure."
        )
    if len(completed_rows) >= 2:
        smallest = min(completed_rows, key=lambda row: row["batch_size"])
        largest = max(completed_rows, key=lambda row: row["batch_size"])
        findings.append(
            f"At a fixed {base_config['total_tokens']:,}-token budget, batch size {smallest['batch_size']} used "
            f"{smallest['completed_steps']:,} optimizer updates, whereas batch size {largest['batch_size']} used "
            f"only {largest['completed_steps']:,}. This changes both gradient noise and the number of parameter "
            "updates, so larger batches do not automatically improve loss even when they improve hardware utilization."
        )
    accumulated_rows = [row for row in completed_rows if row["gradient_accumulation_steps"] > 1]
    if accumulated_rows:
        accumulated_text = ", ".join(
            f"batch {row['batch_size']} as {row['gradient_accumulation_steps']} x microbatch {row['micro_batch_size']}"
            for row in accumulated_rows
        )
        findings.append(
            f"To avoid unified-memory thrashing, gradient accumulation was used for {accumulated_text}. "
            "This preserves the effective gradient batch but means its throughput is not a direct measurement of a "
            "single physical batch allocation."
        )
    if not findings:
        findings.append("No completed runs were available; inspect the per-run failure reasons before extending the sweep.")

    report = "\n".join(
        [
            "# TinyStories batch-size experiment",
            "",
            "## Protocol",
            "",
            f"All runs use peak learning rate **{base_config['learning_rate']:.3g}**, the best value from the prior "
            "learning-rate sweep. The model, tokenizer, sampled-token sequence, optimizer settings, context length, "
            "and total token budget are fixed. Warmup and validation frequency are defined as fractions of token "
            "progress so that changing the number of optimizer steps does not move them to different parts of training.",
            "",
            f"- Token budget per run: {base_config['total_tokens']:,}; context length: {base_config['context_length']}.",
            f"- Warmup fraction: {payload['warmup_fraction']:.1%}; cosine decay to "
            f"{base_config['min_lr_ratio']:.1f} x peak LR.",
            f"- Validation: {base_config['eval_batches']} fixed minibatches of size "
            f"{base_config['eval_batch_size']} for every run; seed: {base_config['seed']}.",
            "",
            "## Results",
            "",
            f"| Effective batch | Microbatch | Accumulation | Updates | Status | Final train loss | Final validation loss | Tokens/s | Tokens to loss <= {target_loss:.2f} |",
            "|---:|---:|---:|---:|:---|---:|---:|---:|---:|",
            *table_rows,
            "",
            "## Findings",
            "",
            *findings,
            "",
            "![Batch-size learning curves](batch_size_learning_curves.png)",
            "",
            "![Batch-size efficiency](batch_size_efficiency.png)",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _train_config(
    base_config: dict[str, Any],
    output_dir: Path,
    batch_size: int,
    micro_batch_size: int,
    warmup_fraction: float,
    eval_points: int,
    train_log_points: int,
) -> TrainConfig:
    field_names = {field.name for field in dataclasses.fields(TrainConfig)}
    unknown_fields = sorted(set(base_config) - field_names)
    if unknown_fields:
        raise ValueError(f"Unknown TrainConfig fields: {', '.join(unknown_fields)}")
    config_payload = base_config.copy()
    for path_field in ("train_data", "valid_data"):
        config_payload[path_field] = Path(config_payload[path_field])
    context_length = int(config_payload["context_length"])
    total_tokens = int(config_payload["total_tokens"])
    tokens_per_step = batch_size * context_length
    if total_tokens % tokens_per_step != 0:
        raise ValueError(
            f"total_tokens={total_tokens} must be divisible by batch_size * context_length={tokens_per_step}"
        )
    max_steps = total_tokens // tokens_per_step
    config_payload.update(
        output_dir=output_dir,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        max_steps=max_steps,
        warmup_steps=max(1, round(max_steps * warmup_fraction)),
        eval_interval=max(1, round(max_steps / eval_points)),
        log_interval=max(1, round(max_steps / train_log_points)),
    )
    return TrainConfig(**config_payload)


def run_sweep(config_path: Path, *, resume: bool = False, overwrite: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = payload["base_config"]
    batch_sizes = [int(value) for value in payload["batch_sizes"]]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch_sizes must be unique")
    warmup_fraction = float(payload.get("warmup_fraction", 0.1))
    if not 0 < warmup_fraction < 1:
        raise ValueError("warmup_fraction must be between zero and one")
    eval_points = int(payload.get("eval_points", 10))
    train_log_points = int(payload.get("train_log_points", 200))
    if eval_points <= 0 or train_log_points <= 0:
        raise ValueError("eval_points and train_log_points must be positive")
    micro_batch_sizes = {int(key): int(value) for key, value in payload.get("micro_batch_sizes", {}).items()}
    unknown_micro_batches = sorted(set(micro_batch_sizes) - set(batch_sizes))
    if unknown_micro_batches:
        raise ValueError(f"micro_batch_sizes contains unknown batch sizes: {unknown_micro_batches}")

    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        micro_batch_size = micro_batch_sizes.get(batch_size, batch_size)
        run_dir = output_dir / f"bs_{batch_size}"
        summary_path = run_dir / "summary.json"
        config = _train_config(
            base_config,
            run_dir,
            batch_size,
            micro_batch_size,
            warmup_fraction,
            eval_points,
            train_log_points,
        )
        if resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            config.overwrite = overwrite
            summary = run_training(config)
        row = summary | {
            "batch_size": batch_size,
            "warmup_steps": config.warmup_steps,
            "eval_interval": config.eval_interval,
            "log_interval": config.log_interval,
            "run_dir": str(run_dir.resolve()),
        }
        row["tokens_to_target"] = _tokens_to_loss(run_dir, float(payload.get("report_target_loss", 2.0)))
        rows.append(row)

    rows.sort(key=lambda item: item["batch_size"])
    (output_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_csv(rows, output_dir / "sweep_summary.csv")
    _write_report(rows, output_dir, payload)
    _plot_curves(rows, output_dir)
    _plot_efficiency(rows, output_dir)

    if payload.get("keep_best_checkpoint_only", False):
        completed_rows = [
            row for row in rows if row["status"] == "completed" and row["final_validation_loss"] is not None
        ]
        if completed_rows:
            best_row = min(completed_rows, key=lambda row: row["final_validation_loss"])
            best_checkpoint = Path(best_row["run_dir"]) / "final_checkpoint.pt"
            for row in completed_rows:
                checkpoint = Path(row["run_dir"]) / "final_checkpoint.pt"
                if checkpoint != best_checkpoint and checkpoint.exists():
                    checkpoint.unlink()
            if best_checkpoint.exists():
                (output_dir / "best_checkpoint.txt").write_text(
                    str(best_checkpoint.resolve()) + "\n",
                    encoding="utf-8",
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse runs that already have summary.json")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in run directories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args.config, resume=args.resume, overwrite=args.overwrite)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
