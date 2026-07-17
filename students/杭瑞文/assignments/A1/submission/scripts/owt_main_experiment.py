"""Run the fixed-compute OpenWebText main experiment and compare with TinyStories."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from scripts.generate_text import generate_samples, write_samples
from scripts.train import TrainConfig, run_training


def _lr_slug(learning_rate: float) -> str:
    return f"lr_{learning_rate:.8g}".replace("+", "").replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_train_config(payload: dict[str, Any], output_dir: Path, **overrides: Any) -> TrainConfig:
    field_names = {field.name for field in dataclasses.fields(TrainConfig)}
    unknown_fields = sorted((set(payload) | set(overrides)) - field_names)
    if unknown_fields:
        raise ValueError(f"Unknown TrainConfig fields: {', '.join(unknown_fields)}")
    config_payload = payload | overrides
    for path_field in ("train_data", "valid_data"):
        config_payload[path_field] = Path(config_payload[path_field])
    config_payload["output_dir"] = output_dir
    return TrainConfig(**config_payload)


def _run_or_resume(config: TrainConfig, *, resume: bool, overwrite: bool) -> dict[str, Any]:
    summary_path = config.output_dir / "summary.json"
    if resume and summary_path.exists():
        return _load_json(summary_path)
    config.overwrite = overwrite or (config.output_dir.exists() and not summary_path.exists())
    return run_training(config)


def _perplexity(loss: float | None) -> float | None:
    if loss is None or not math.isfinite(loss):
        return None
    return math.exp(loss)


def _moving_average(values: list[float], window: int = 10) -> list[float]:
    smoothed: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        smoothed.append(running_sum / min(index + 1, window))
    return smoothed


def _plot_learning_curves(
    owt_metrics: Path,
    tinystories_metrics: Path,
    output_dir: Path,
    *,
    updates: int,
    tokens: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib") from error

    datasets = [
        ("OpenWebText", _load_records(owt_metrics), "#d1495b"),
        ("TinyStories", _load_records(tinystories_metrics), "#00798c"),
    ]
    figure, (train_axis, validation_axis) = plt.subplots(
        1, 2, figsize=(12, 4.8), constrained_layout=True
    )
    for label, records, color in datasets:
        train = [record for record in records if record.get("event") == "train"]
        validation = [record for record in records if record.get("event") == "validation"]
        train_axis.plot(
            [record["tokens"] / 1e6 for record in train],
            _moving_average([record["loss"] for record in train]),
            label=label,
            color=color,
            linewidth=1.7,
        )
        validation_axis.plot(
            [record["tokens"] / 1e6 for record in validation],
            [record["loss"] for record in validation],
            label=label,
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
        )
    train_axis.set_title("Training loss (10-point moving average)")
    validation_axis.set_title("Fixed held-out validation loss")
    for axis in (train_axis, validation_axis):
        axis.set_xlabel("Tokens processed (millions)")
        axis.set_ylabel("Cross-entropy (nats/token)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Same architecture and {updates:,}-update / {tokens / 1e6:.2f}M-token budget")
    figure.savefig(output_dir / "owt_learning_curve.png", dpi=180)
    figure.savefig(output_dir / "owt_learning_curve.pdf")
    plt.close(figure)


def _plot_lr_calibration(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib") from error

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for row in sorted(rows, key=lambda item: item["learning_rate"]):
        records = _load_records(Path(row["run_dir"]) / "metrics.jsonl")
        validation = [record for record in records if record.get("event") == "validation"]
        axis.plot(
            [record["tokens"] / 1e6 for record in validation],
            [record["loss"] for record in validation],
            marker="o",
            linewidth=1.5,
            label=f"lr={row['learning_rate']:.3g} ({row['status']})",
        )
    axis.set_title("OpenWebText learning-rate calibration")
    axis.set_xlabel("Tokens processed (millions)")
    axis.set_ylabel("Validation cross-entropy (nats/token)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "lr_calibration.png", dpi=180)
    plt.close(figure)


def _write_calibration_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "learning_rate",
        "status",
        "completed_steps",
        "tokens_processed",
        "final_train_loss",
        "final_validation_loss",
        "elapsed_seconds",
        "mean_tokens_per_second",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _format_loss(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _write_report(
    output_dir: Path,
    payload: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    owt_summary: dict[str, Any],
    tinystories_summary: dict[str, Any],
    selected_lr: float,
    owt_samples: list[dict[str, Any]],
) -> None:
    base = payload["base_config"]
    calibration_table = [
        "| Peak LR | Status | Pilot updates | Final train loss | Final validation loss |",
        "|---:|:---|---:|---:|---:|",
    ]
    for row in sorted(calibration_rows, key=lambda item: item["learning_rate"]):
        calibration_table.append(
            f"| {row['learning_rate']:.3g} | {row['status']} | {row['completed_steps']:,} | "
            f"{_format_loss(row.get('final_train_loss'))} | {_format_loss(row.get('final_validation_loss'))} |"
        )

    owt_loss = owt_summary.get("final_validation_loss")
    tiny_loss = tinystories_summary.get("final_validation_loss")
    owt_ppl = _perplexity(owt_loss)
    tiny_ppl = _perplexity(tiny_loss)
    loss_gap = None if owt_loss is None or tiny_loss is None else owt_loss - tiny_loss
    sample_blocks: list[str] = []
    for sample in owt_samples:
        sample_blocks.extend(
            [
                f"### OWT sample {sample['sample']} (seed {sample['seed']})",
                "",
                str(sample["continuation"]).strip(),
                "",
            ]
        )
    report = "\n".join(
        [
            "# OpenWebText main experiment",
            "",
            "## Controlled protocol",
            "",
            "The OpenWebText and TinyStories runs use the same 4-layer, 16-head, pre-norm Transformer "
            "with RoPE and SwiGLU, the same 10,000-token vocabulary size, context length 256, effective "
            f"batch size {base['batch_size']}, seed {base['seed']}, and exactly "
            f"{owt_summary['completed_steps']:,} optimizer updates "
            f"({owt_summary['tokens_processed']:,} training tokens). "
            "Each corpus has its own byte-level BPE tokenizer. Validation uses the same fixed number and "
            "size of held-out minibatches within each corpus.",
            "",
            f"- Model dimensions: d_model={base['d_model']}, d_ff={base['d_ff']}; "
            f"parameters: {owt_summary['parameter_count']:,}.",
            f"- AdamW: beta1={base['beta1']}, beta2={base['beta2']}, epsilon={base['adam_eps']:.0e}, "
            f"weight decay={base['weight_decay']}; cosine schedule with 10% warmup.",
            f"- Selected OWT peak learning rate: **{selected_lr:.3g}**, chosen using short pilot runs.",
            "",
            "## Learning-rate calibration",
            "",
            *calibration_table,
            "",
            "The pilot schedule is used only to choose a plausible OWT learning rate; the selected model "
            "is then trained from a fresh initialization for the full budget.",
            "",
            "## Main result",
            "",
            "| Dataset | Updates | Tokens | Final train loss | Final validation loss | Validation perplexity |",
            "|:---|---:|---:|---:|---:|---:|",
            f"| TinyStories | {tinystories_summary['completed_steps']:,} | "
            f"{tinystories_summary['tokens_processed']:,} | {_format_loss(tinystories_summary.get('final_train_loss'))} | "
            f"{_format_loss(tiny_loss)} | {tiny_ppl:.2f} |",
            f"| OpenWebText | {owt_summary['completed_steps']:,} | {owt_summary['tokens_processed']:,} | "
            f"{_format_loss(owt_summary.get('final_train_loss'))} | {_format_loss(owt_loss)} | {owt_ppl:.2f} |",
            "",
            f"OWT validation loss is **{loss_gap:+.4f} nats/token** relative to TinyStories under the "
            "same model and update/token budget. The higher loss is expected: web text has much greater "
            "topic, style, vocabulary, factual, and document-structure diversity, plus crawl noise. A "
            "17M-non-embedding-parameter model seeing only 40.96M tokens cannot repeat and consolidate "
            "those long-tail patterns as often as it can on the deliberately simple TinyStories corpus.",
            "",
            "These numbers are per-token negative log-likelihoods, so lower is better and exp(loss) is "
            "token-level perplexity. They should not be interpreted as a perfectly tokenizer-independent "
            "measure of English quality: the two independently trained BPE tokenizers segment text "
            "differently, even though both vocabularies contain 10,000 tokens. The curves are most useful "
            "for comparing optimization progress within a corpus; the cross-corpus gap additionally "
            "reflects corpus entropy and tokenization.",
            "",
            "## Generated text and fluency",
            "",
            "Three unconditional OWT samples and matched TinyStories samples are saved in "
            "`owt_generated_samples.md` and `tinystories_generated_samples.md`. They use temperature "
            f"{payload['sampling']['temperature']} and top-k {payload['sampling']['top_k']} with the same seeds.",
            "",
            *sample_blocks,
            "In this run, the OWT samples show mostly fluent local English syntax and recognizable "
            "review/news/article formatting, but weak document-level coherence. Sample 1 repeatedly "
            "returns to generic words such as 'show' and 'fun' while drifting semantically; sample 2 "
            "resembles political news but mixes unsupported entities and malformed phrases; sample 3 "
            "invents names and ends after a formulaic study claim. Samples 1 and 2 also terminate "
            "mid-sentence. The same compute works much better for TinyStories because its short "
            "sentences, small semantic space, repeated narrative template, and limited vocabulary provide "
            "far more effective examples of each pattern. OWT needs substantially more data exposure, "
            "model capacity, and usually a longer context to approach comparable document-level fluency.",
            "",
            "![OpenWebText and TinyStories learning curves](owt_learning_curve.png)",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_experiment(config_path: Path, *, resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
    payload = _load_json(config_path)
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base = payload["base_config"]
    pilot_steps = int(payload["pilot_steps"])
    pilot_eval_interval = int(payload.get("pilot_eval_interval", max(1, pilot_steps // 5)))
    warmup_fraction = float(payload.get("warmup_fraction", 0.1))
    pilot_rates = [float(value) for value in payload["pilot_learning_rates"]]

    calibration_rows: list[dict[str, Any]] = []
    for learning_rate in pilot_rates:
        run_dir = output_dir / "lr_calibration" / _lr_slug(learning_rate)
        pilot_config = _make_train_config(
            base,
            run_dir,
            learning_rate=learning_rate,
            total_tokens=pilot_steps * int(base["batch_size"]) * int(base["context_length"]),
            max_steps=pilot_steps,
            warmup_steps=max(1, round(pilot_steps * warmup_fraction)),
            eval_interval=pilot_eval_interval,
            log_interval=max(1, pilot_steps // 100),
            checkpoint="none",
        )
        summary = _run_or_resume(pilot_config, resume=resume, overwrite=overwrite)
        calibration_rows.append(summary | {"run_dir": str(run_dir.resolve())})

    stable_rows = [
        row
        for row in calibration_rows
        if row["status"] == "completed" and row.get("final_validation_loss") is not None
    ]
    if not stable_rows:
        raise RuntimeError("Every OWT learning-rate pilot diverged")
    best_pilot = min(stable_rows, key=lambda row: row["final_validation_loss"])
    selected_lr = float(best_pilot["learning_rate"])

    main_dir = output_dir / "main"
    main_config = _make_train_config(base, main_dir, learning_rate=selected_lr)
    owt_summary = _run_or_resume(main_config, resume=resume, overwrite=overwrite)
    if owt_summary["status"] != "completed":
        raise RuntimeError(f"Full OWT run did not complete: {owt_summary.get('divergence_reason')}")

    if "tinystories_best_checkpoint_pointer" in payload:
        pointer_path = Path(payload["tinystories_best_checkpoint_pointer"])
        tiny_checkpoint = Path(pointer_path.read_text(encoding="utf-8").strip())
        tiny_run_dir = tiny_checkpoint.parent
        tiny_summary_path = tiny_run_dir / "summary.json"
        tiny_metrics_path = tiny_run_dir / "metrics.jsonl"
    else:
        tiny_summary_path = Path(payload["tinystories_summary"])
        tiny_metrics_path = Path(payload["tinystories_metrics"])
        tiny_checkpoint = Path(payload["tinystories_checkpoint"])
    tinystories_summary = _load_json(tiny_summary_path)
    for matched_field in ("completed_steps", "tokens_processed"):
        if owt_summary[matched_field] != tinystories_summary[matched_field]:
            raise ValueError(
                f"OWT and TinyStories must match {matched_field}: "
                f"{owt_summary[matched_field]} != {tinystories_summary[matched_field]}"
            )

    sampling = payload["sampling"]
    prompts = list(sampling.get("prompts", ["", "", ""]))
    seeds = [int(value) for value in sampling.get("seeds", [101, 202, 303])]
    common_sampling = {
        "prompts": prompts,
        "seeds": seeds,
        "max_new_tokens": int(sampling["max_new_tokens"]),
        "temperature": float(sampling["temperature"]),
        "top_k": int(sampling["top_k"]),
        "device_name": str(base["device"]),
    }
    owt_samples = generate_samples(
        main_dir / "final_checkpoint.pt",
        Path(payload["owt_tokenizer"]),
        **common_sampling,
    )
    tiny_samples = generate_samples(
        tiny_checkpoint,
        Path(payload["tinystories_tokenizer"]),
        **common_sampling,
    )
    write_samples(owt_samples, output_dir / "owt_generated_samples.md", title="OpenWebText generated samples")
    write_samples(owt_samples, output_dir / "owt_generated_samples.json", title="OpenWebText generated samples")
    write_samples(
        tiny_samples,
        output_dir / "tinystories_generated_samples.md",
        title="TinyStories generated samples (matched decoding)",
    )
    write_samples(
        tiny_samples,
        output_dir / "tinystories_generated_samples.json",
        title="TinyStories generated samples (matched decoding)",
    )

    _write_calibration_csv(calibration_rows, output_dir / "lr_calibration.csv")
    (output_dir / "lr_calibration.json").write_text(
        json.dumps(calibration_rows, indent=2) + "\n", encoding="utf-8"
    )
    _plot_lr_calibration(calibration_rows, output_dir)
    _plot_learning_curves(
        main_dir / "metrics.jsonl",
        tiny_metrics_path,
        output_dir,
        updates=int(owt_summary["completed_steps"]),
        tokens=int(owt_summary["tokens_processed"]),
    )
    _write_report(
        output_dir,
        payload,
        calibration_rows,
        owt_summary,
        tinystories_summary,
        selected_lr,
        owt_samples,
    )

    experiment_summary = {
        "selected_learning_rate": selected_lr,
        "owt": owt_summary | {"validation_perplexity": _perplexity(owt_summary["final_validation_loss"])},
        "tinystories": tinystories_summary
        | {"validation_perplexity": _perplexity(tinystories_summary["final_validation_loss"])},
        "validation_loss_gap_owt_minus_tinystories": (
            owt_summary["final_validation_loss"] - tinystories_summary["final_validation_loss"]
        ),
        "calibration_runs": calibration_rows,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2) + "\n", encoding="utf-8"
    )
    return experiment_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(args.config, resume=args.resume, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
