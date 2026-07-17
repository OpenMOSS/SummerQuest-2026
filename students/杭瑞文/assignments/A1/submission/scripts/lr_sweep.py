"""Run, summarize, and plot a reproducible learning-rate sweep."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from scripts.train import TrainConfig, run_training


def _lr_slug(learning_rate: float) -> str:
    return f"lr_{learning_rate:.8g}".replace("+", "").replace(".", "p")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "learning_rate",
        "status",
        "completed_steps",
        "tokens_processed",
        "initial_validation_loss",
        "final_train_loss",
        "final_validation_loss",
        "best_validation_loss",
        "elapsed_seconds",
        "mean_tokens_per_second",
        "effective_divergence_threshold",
        "divergence_reason",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def _tokens_to_loss(run_dir: Path, target_loss: float) -> int | None:
    for record in _load_records(run_dir / "metrics.jsonl"):
        if record.get("event") == "validation" and record["loss"] <= target_loss:
            return int(record["tokens"])
    return None


def _format_loss(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _write_report(rows: list[dict[str, Any]], output_dir: Path, sweep_config: dict[str, Any]) -> None:
    base_config = sweep_config["base_config"]
    stable_rows = [row for row in rows if row["status"] == "completed" and row["final_validation_loss"] is not None]
    diverged_rows = [row for row in rows if row["status"] == "diverged"]
    best_row = min(stable_rows, key=lambda row: row["final_validation_loss"]) if stable_rows else None
    target_loss = float(sweep_config.get("report_target_loss", 2.0))

    table_rows: list[str] = []
    for row in sorted(rows, key=lambda item: item["learning_rate"]):
        tokens_to_target = None
        if target_loss is not None:
            tokens_to_target = _tokens_to_loss(Path(row["run_dir"]), target_loss)
        table_rows.append(
            "| {lr:.3g} | {status} | {train} | {valid} | {best} | {tokens} |".format(
                lr=row["learning_rate"],
                status=row["status"],
                train=_format_loss(row.get("final_train_loss")),
                valid=_format_loss(row.get("final_validation_loss")),
                best=_format_loss(row.get("best_validation_loss")) if row["status"] == "completed" else "-",
                tokens="-" if tokens_to_target is None else f"{tokens_to_target:,}",
            )
        )

    analysis: list[str] = []
    if best_row is not None:
        analysis.append(
            f"The lowest final validation loss was **{best_row['final_validation_loss']:.4f}** at "
            f"learning rate **{best_row['learning_rate']:.3g}**."
        )
        higher_stable = [
            row
            for row in stable_rows
            if row["learning_rate"] > best_row["learning_rate"]
            and row["final_validation_loss"] > best_row["final_validation_loss"]
        ]
        if higher_stable:
            first_worse = min(higher_stable, key=lambda row: row["learning_rate"])
            analysis.append(
                f"Increasing the rate to **{first_worse['learning_rate']:.3g}** remained numerically stable, "
                f"but worsened final validation loss to **{first_worse['final_validation_loss']:.4f}**. "
                "The useful optimum therefore occurs before the numerical stability boundary."
            )
    if stable_rows and diverged_rows:
        largest_stable = max(row["learning_rate"] for row in stable_rows)
        larger_diverged_rows = [row for row in diverged_rows if row["learning_rate"] > largest_stable]
        larger_diverged = [row["learning_rate"] for row in larger_diverged_rows]
        if larger_diverged:
            smallest_diverged = min(larger_diverged)
            analysis.append(
                f"For this schedule and seed, the observed stability edge is bracketed by "
                f"**{largest_stable:.3g}** (completed) and **{smallest_diverged:.3g}** (diverged)."
            )
            divergent_row = min(larger_diverged_rows, key=lambda row: row["learning_rate"])
            analysis.append(
                f"The divergent run stopped after **{divergent_row['tokens_processed']:,} tokens**: "
                f"{divergent_row['divergence_reason']}."
            )
    convergence_rows = []
    for row in stable_rows:
        tokens = _tokens_to_loss(Path(row["run_dir"]), target_loss)
        if tokens is not None:
            convergence_rows.append((tokens, row["learning_rate"]))
    if convergence_rows:
        first_tokens = min(tokens for tokens, _ in convergence_rows)
        fastest_rates = sorted(learning_rate for tokens, learning_rate in convergence_rows if tokens == first_tokens)
        rate_text = " and ".join(f"**{learning_rate:.3g}**" for learning_rate in fastest_rates)
        analysis.append(
            f"At the configured target loss of **{target_loss:.2f}**, learning rate {rate_text} reached "
            f"the target at the first logged crossing, **{first_tokens:,} tokens**."
        )
    if not analysis:
        analysis.append("No stable completed run was available, so the sweep must be extended or debugged.")

    report = "\n".join(
        [
            "# TinyStories learning-rate sweep",
            "",
            "## Search strategy",
            "",
            "All runs keep the data, model, batch size, token budget, AdamW settings, cosine schedule, "
            "validation batches, and random seed fixed; only the peak learning rate changes. The rates "
            "are spaced approximately logarithmically, with an additional high-rate probe when necessary "
            "to locate a divergent run. Training loss is logged frequently and validation loss uses the "
            "same deterministic held-out minibatches for every run.",
            "",
            "## Experimental setup",
            "",
            f"- Device: `{base_config['device']}`; seed: `{base_config['seed']}`.",
            f"- Model: {base_config['num_layers']} layers, {base_config['num_heads']} heads, "
            f"d_model={base_config['d_model']}, d_ff={base_config['d_ff']}, "
            f"context length={base_config['context_length']}, vocabulary={base_config['vocab_size']:,}.",
            f"- Budget per run: {base_config['total_tokens']:,} tokens "
            f"(batch size {base_config['batch_size']}); warmup: {base_config['warmup_steps']} steps; "
            f"cosine decay to {base_config['min_lr_ratio']:.1f} x peak LR.",
            f"- AdamW: beta1={base_config['beta1']}, beta2={base_config['beta2']}, "
            f"epsilon={base_config['adam_eps']:.0e}, weight decay={base_config['weight_decay']}, "
            f"gradient clipping={base_config['grad_clip']}.",
            "",
            "## (a) Sweep results",
            "",
            f"| Peak LR | Status | Final train loss | Final validation loss | Best validation loss | Tokens to loss <= {target_loss:.2f} |",
            "|---:|:---|---:|---:|---:|---:|",
            *table_rows,
            "",
            "## (b) Stability edge and convergence",
            "",
            *analysis,
            "",
            "The stability boundary is empirical rather than universal: it depends on the warmup, batch "
            "size, optimizer, initialization, clipping threshold, and seed. A rate just below the boundary "
            "is useful only when its validation curve also converges faster or lower; proximity to divergence "
            "alone is not evidence that it is best.",
            "",
            "![Learning-rate curves](learning_rate_curves.png)",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _smoothed(values: list[float], window: int = 10) -> list[float]:
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


def _plot_curves(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib; run `uv add matplotlib` and retry") from error

    initial_losses = [
        row["initial_validation_loss"]
        for row in rows
        if row.get("initial_validation_loss") is not None and math.isfinite(row["initial_validation_loss"])
    ]
    loss_ceiling = max(5.0, 1.05 * max(initial_losses, default=5.0))
    figure, (train_axis, validation_axis) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for row in sorted(rows, key=lambda item: item["learning_rate"]):
        records = _load_records(Path(row["run_dir"]) / "metrics.jsonl")
        train_records = [record for record in records if record.get("event") == "train"]
        validation_records = [record for record in records if record.get("event") == "validation"]
        label = f"lr={row['learning_rate']:.3g} ({row['status']})"
        if train_records:
            tokens = [record["tokens"] / 1e6 for record in train_records]
            losses = _smoothed([record["loss"] for record in train_records])
            train_axis.plot(tokens, losses, linewidth=1.5, label=label)
            if row["status"] == "diverged":
                divergence_loss = row.get("final_train_loss")
                if divergence_loss is not None and math.isfinite(divergence_loss):
                    marker_loss = min(divergence_loss, loss_ceiling)
                    marker_tokens = row["tokens_processed"] / 1e6
                    train_axis.scatter(marker_tokens, marker_loss, marker="x", s=70, color="black", zorder=5)
                    train_axis.annotate(
                        "diverged",
                        (marker_tokens, marker_loss),
                        xytext=(5, -12),
                        textcoords="offset points",
                        fontsize=8,
                    )
        if validation_records:
            tokens = [record["tokens"] / 1e6 for record in validation_records]
            losses = [record["loss"] for record in validation_records]
            validation_axis.plot(tokens, losses, marker="o", markersize=3, linewidth=1.5, label=label)

    train_axis.set_title("Training loss (10-point moving average)")
    validation_axis.set_title("Held-out validation loss")
    for axis in (train_axis, validation_axis):
        axis.set_xlabel("Tokens processed (millions)")
        axis.set_ylabel("Per-token cross-entropy")
        axis.set_ylim(bottom=1.0, top=loss_ceiling)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / "learning_rate_curves.png", dpi=180)
    figure.savefig(output_dir / "learning_rate_curves.pdf")
    plt.close(figure)


def _train_config_from_dict(payload: dict[str, Any], output_dir: Path, learning_rate: float) -> TrainConfig:
    field_names = {field.name for field in dataclasses.fields(TrainConfig)}
    unknown_fields = sorted(set(payload) - field_names)
    if unknown_fields:
        raise ValueError(f"Unknown TrainConfig fields: {', '.join(unknown_fields)}")
    config_payload = payload.copy()
    for path_field in ("train_data", "valid_data"):
        config_payload[path_field] = Path(config_payload[path_field])
    config_payload["output_dir"] = output_dir
    config_payload["learning_rate"] = learning_rate
    return TrainConfig(**config_payload)


def run_sweep(config_path: Path, *, resume: bool = False, overwrite: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = payload["base_config"]
    learning_rates = [float(value) for value in payload["learning_rates"]]
    if not learning_rates:
        raise ValueError("learning_rates cannot be empty")
    if any(not math.isfinite(value) or value <= 0 for value in learning_rates):
        raise ValueError("learning_rates must contain only positive finite values")

    ensure_divergent = bool(payload.get("ensure_divergent", True))
    divergence_multiplier = float(payload.get("divergence_multiplier", 3.0))
    max_extra_divergence_probes = int(payload.get("max_extra_divergence_probes", 3))
    rows: list[dict[str, Any]] = []
    pending_rates = list(dict.fromkeys(learning_rates))
    extra_probes = 0
    while pending_rates:
        learning_rate = pending_rates.pop(0)
        run_dir = output_dir / _lr_slug(learning_rate)
        summary_path = run_dir / "summary.json"
        if resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            config = _train_config_from_dict(base_config, run_dir, learning_rate)
            config.overwrite = overwrite
            summary = run_training(config)
        row = summary | {"run_dir": str(run_dir.resolve())}
        rows.append(row)

        if not pending_rates and ensure_divergent and not any(item["status"] == "diverged" for item in rows):
            if extra_probes < max_extra_divergence_probes:
                next_rate = max(item["learning_rate"] for item in rows) * divergence_multiplier
                pending_rates.append(next_rate)
                extra_probes += 1

    rows.sort(key=lambda item: item["learning_rate"])
    (output_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_csv(rows, output_dir / "sweep_summary.csv")
    _write_report(rows, output_dir, payload)
    _plot_curves(rows, output_dir)

    if payload.get("keep_best_checkpoint_only", False):
        stable_rows = [row for row in rows if row["status"] == "completed" and row["final_validation_loss"] is not None]
        if stable_rows:
            best_row = min(stable_rows, key=lambda row: row["final_validation_loss"])
            best_checkpoint = Path(best_row["run_dir"]) / "final_checkpoint.pt"
            for row in stable_rows:
                checkpoint = Path(row["run_dir"]) / "final_checkpoint.pt"
                if checkpoint != best_checkpoint and checkpoint.exists():
                    checkpoint.unlink()
            if best_checkpoint.exists():
                (output_dir / "best_checkpoint.txt").write_text(str(best_checkpoint.resolve()) + "\n", encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse runs that already have summary.json")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in existing run directories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args.config, resume=args.resume, overwrite=args.overwrite)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
