from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.config import load_json_config
from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW, clip_gradients, get_lr_cosine_schedule
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer language model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Show step and validation progress on stderr.")
    return parser.parse_args()


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def build_model(config: dict[str, Any], device: torch.device) -> TransformerLM:
    model = config["model"]
    ablation = config.get("ablation", {})
    return TransformerLM(
        vocab_size=model["vocab_size"],
        context_length=model["context_length"],
        d_model=model["d_model"],
        num_layers=model["num_layers"],
        num_heads=model["num_heads"],
        d_ff=model["d_ff"],
        rope_theta=model.get("rope_theta", 10_000.0),
        remove_rmsnorm=ablation.get("remove_rmsnorm", False),
        use_post_norm=ablation.get("use_post_norm", False),
        remove_rope=ablation.get("remove_rope", False),
        ffn_type=ablation.get("ffn_type"),
        device=device,
    )


def autocast_context(device: torch.device, precision: str | None):
    if device.type != "cuda" or precision is None:
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def read_training_log(path: Path, *, max_step: int | None = None) -> list[dict[str, Any]]:
    """Read complete JSONL records, optionally stopping at an absolute step."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
        step = int(record["step"])
        if max_step is None or step <= max_step:
            records.append(record)
    return records


def write_training_log(path: Path, records: list[dict[str, Any]]) -> None:
    """Rewrite a resumed log so it contains no records beyond the resume point."""
    contents = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(contents, encoding="utf-8")


def best_validation_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in records if "val_loss" in record]
    return min(candidates, key=lambda record: float(record["val_loss"]), default=None)


def save_best_checkpoint(
    model: TransformerLM,
    optimizer: AdamW,
    *,
    step: int,
    val_loss: float,
    output_dir: Path,
) -> Path:
    """Atomically replace the single best checkpoint and its small metadata file."""
    checkpoint_path = output_dir / "checkpoint_best.pt"
    temporary_path = output_dir / ".checkpoint_best.pt.tmp"
    save_checkpoint(model, optimizer, step, temporary_path)
    temporary_path.replace(checkpoint_path)
    metadata = {
        "best_val_loss": val_loss,
        "best_step": step,
        "best_checkpoint": str(checkpoint_path),
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checkpoint_path


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
    precision: str | None,
    progress: bool = False,
) -> float:
    model.eval()
    losses = []
    for _ in tqdm(
        range(eval_batches),
        desc="Validation",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    ):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        with autocast_context(device, precision):
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    device = resolve_device(config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    training = config["training"]
    optimizer_config = config["optimizer"]
    model_config = config["model"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    train_data = np.load(config["train_data"], mmap_mode="r")
    validation_data = np.load(config["validation_data"], mmap_mode="r")
    model = build_model(config, device)
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config["max_lr"],
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0.01),
    )

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(args.resume, model, optimizer)

    total_steps = 2 if args.dry_run else int(training["steps"])
    if start_iteration > total_steps:
        raise ValueError(f"resume checkpoint is at step {start_iteration}, beyond configured total steps {total_steps}")
    eval_interval = 1 if args.dry_run else int(training["eval_interval"])
    checkpoint_interval = 1 if args.dry_run else int(training["checkpoint_interval"])
    log_interval = 1 if args.dry_run else int(training.get("log_interval", 1))
    save_periodic_checkpoints = bool(training.get("save_checkpoints", True))
    save_best = bool(training.get("save_best", True))
    precision = training.get("mixed_precision")
    batch_size = int(training["batch_size"])
    context_length = int(model_config["context_length"])
    log_path = output_dir / "train.jsonl"
    if args.resume:
        historical_records = read_training_log(log_path, max_step=start_iteration)
        write_training_log(log_path, historical_records)
    else:
        historical_records = read_training_log(log_path)
        if historical_records:
            raise FileExistsError(f"{log_path} already contains training records; use --resume or a new output_dir")

    last_record: dict[str, Any] | None = historical_records[-1] if historical_records else None
    prior_best = best_validation_record(historical_records)
    best_val_loss = float(prior_best["val_loss"]) if prior_best is not None else math.inf
    best_step = int(prior_best["step"]) if prior_best is not None else None
    best_checkpoint: Path | None = None
    best_metadata_path = output_dir / "best_checkpoint.json"
    if args.resume and best_metadata_path.exists():
        best_metadata = json.loads(best_metadata_path.read_text(encoding="utf-8"))
        metadata_step = int(best_metadata["best_step"])
        metadata_checkpoint = Path(best_metadata["best_checkpoint"])
        if metadata_step <= start_iteration and metadata_checkpoint.exists():
            best_val_loss = float(best_metadata["best_val_loss"])
            best_step = metadata_step
            best_checkpoint = metadata_checkpoint
        elif metadata_step > start_iteration:
            # The single best file belongs to the abandoned suffix of this run.
            # Remove it rather than silently presenting a future model as the resumed run's best.
            metadata_checkpoint.unlink(missing_ok=True)
            best_metadata_path.unlink(missing_ok=True)

    elapsed_offset = max(
        (float(record.get("wall_clock_sec", 0.0)) for record in historical_records),
        default=0.0,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()

    with log_path.open("a", encoding="utf-8") as log_file:
        training_steps = tqdm(
            range(start_iteration, total_steps),
            total=total_steps,
            initial=start_iteration,
            desc="LM training",
            unit="step",
            dynamic_ncols=True,
            disable=not args.progress,
        )
        for iteration in training_steps:
            learning_rate = get_lr_cosine_schedule(
                iteration,
                optimizer_config["max_lr"],
                optimizer_config["min_lr"],
                optimizer_config["warmup_iters"],
                optimizer_config["cosine_cycle_iters"],
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            inputs, targets = get_batch(train_data, batch_size, context_length, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
            loss.backward()
            clip_gradients(model.parameters(), training["max_grad_norm"])
            optimizer.step()

            step = iteration + 1
            record: dict[str, Any] = {
                "run_name": config["run_name"],
                "step": step,
                "processed_tokens": step * batch_size * context_length,
                "wall_clock_sec": elapsed_offset + time.perf_counter() - started,
                "train_loss": float(loss.item()),
                "lr": learning_rate,
            }
            if step % eval_interval == 0 or step == total_steps:
                record["val_loss"] = evaluate(
                    model,
                    validation_data,
                    batch_size=batch_size,
                    context_length=context_length,
                    eval_batches=int(training["eval_batches"]),
                    device=device,
                    precision=precision,
                    progress=args.progress,
                )
            postfix = {
                "loss": f"{record['train_loss']:.4f}",
                "lr": f"{learning_rate:.2e}",
            }
            if "val_loss" in record:
                postfix["val"] = f"{record['val_loss']:.4f}"
                if float(record["val_loss"]) < best_val_loss:
                    best_val_loss = float(record["val_loss"])
                    best_step = step
                    if save_best:
                        best_checkpoint = save_best_checkpoint(
                            model,
                            optimizer,
                            step=step,
                            val_loss=best_val_loss,
                            output_dir=output_dir,
                        )
            training_steps.set_postfix(postfix)
            if step % log_interval == 0 or "val_loss" in record:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                print(json.dumps(record, ensure_ascii=False))
            if save_periodic_checkpoints and (step % checkpoint_interval == 0 or step == total_steps):
                save_checkpoint(model, optimizer, step, output_dir / f"checkpoint_{step:08d}.pt")
            last_record = record
        training_steps.close()

    session_wall_clock_sec = time.perf_counter() - started
    session_processed_tokens = (total_steps - start_iteration) * batch_size * context_length
    summary = {
        "run_name": config["run_name"],
        "status": "completed",
        "steps": total_steps,
        "batch_size": batch_size,
        "context_length": context_length,
        "processed_tokens": total_steps * batch_size * context_length,
        "wall_clock_sec": elapsed_offset + session_wall_clock_sec,
        "session_wall_clock_sec": session_wall_clock_sec,
        "tokens_per_sec": session_processed_tokens / session_wall_clock_sec,
        "peak_gpu_memory_bytes": (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
        "final_train_loss": last_record.get("train_loss") if last_record else None,
        "final_val_loss": last_record.get("val_loss") if last_record else None,
        "best_val_loss": None if best_step is None else best_val_loss,
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "start_step": start_iteration,
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "model": model_config,
        "optimizer": optimizer_config,
        "ablation": config.get("ablation", {}),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
