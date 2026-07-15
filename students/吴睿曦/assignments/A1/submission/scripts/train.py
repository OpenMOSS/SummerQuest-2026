import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    clip_gradients,
    cosine_schedule,
    cross_entropy,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)


def autocast_context(config, device):
    amp_dtype = config.get("amp_dtype")
    if device.type != "cuda" or amp_dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be null, float16, or bfloat16")
    return torch.autocast(device_type="cuda", dtype=getattr(torch, amp_dtype))


def evaluate(model, data, config, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config["eval_batches"]):
            x, y = get_batch(data, config["batch_size"], config["context_length"], device)
            with autocast_context(config, torch.device(device)):
                losses.append(cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def _nonnegative_finite(value, field: str, source: Path) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{source}: {field} must be finite and non-negative")
    return number


def _nonnegative_int(value, field: str, source: Path) -> int:
    number = int(value)
    if number < 0 or number != float(value):
        raise ValueError(f"{source}: {field} must be a non-negative integer")
    return number


def load_resume_metrics(
    log_path: Path,
    summary_path: Path,
    start_iteration: int,
) -> tuple[float, int, float | None]:
    """Load exact cumulative metrics at a checkpoint boundary before appending logs."""
    boundary_record = None
    if log_path.is_file():
        with log_path.open(encoding="utf-8") as log:
            for line_number, line in enumerate(log, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or "step" not in record:
                    raise ValueError(f"{log_path}:{line_number}: expected an object with a step")
                record_step = _nonnegative_int(record["step"], "step", log_path)
                if record_step > start_iteration:
                    raise ValueError(
                        f"{log_path}:{line_number}: log step {record_step} is newer than "
                        f"checkpoint step {start_iteration}; use a matching checkpoint or output directory"
                    )
                if record_step == start_iteration:
                    boundary_record = record

    boundary_summary = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or "completed_steps" not in summary:
            raise ValueError(f"{summary_path}: expected an object with completed_steps")
        summary_step = _nonnegative_int(summary["completed_steps"], "completed_steps", summary_path)
        if summary_step > start_iteration:
            raise ValueError(
                f"{summary_path}: summary step {summary_step} is newer than checkpoint step "
                f"{start_iteration}; use a matching checkpoint or output directory"
            )
        if summary_step == start_iteration:
            boundary_summary = summary

    if boundary_summary is None and boundary_record is None:
        raise ValueError(
            f"cannot recover cumulative metrics for checkpoint step {start_iteration}; "
            "the existing summary or log must contain that exact step"
        )

    source = summary_path if boundary_summary is not None else log_path
    history = boundary_summary if boundary_summary is not None else boundary_record
    time_key = "total_training_sec" if boundary_summary is not None else "wall_clock_sec"
    tokens_key = "total_processed_tokens" if boundary_summary is not None else "processed_tokens"
    history_sec = _nonnegative_finite(history[time_key], time_key, source)
    history_tokens = _nonnegative_int(history[tokens_key], tokens_key, source)
    final_val_loss = history.get("final_val_loss" if boundary_summary is not None else "val_loss")

    if boundary_record is not None:
        record_sec = _nonnegative_finite(boundary_record["wall_clock_sec"], "wall_clock_sec", log_path)
        record_tokens = _nonnegative_int(boundary_record["processed_tokens"], "processed_tokens", log_path)
        if boundary_summary is not None and record_tokens != history_tokens:
            raise ValueError(
                f"checkpoint-boundary token counts disagree between {log_path} and {summary_path}"
            )
        history_sec = max(history_sec, record_sec)
        if final_val_loss is None:
            final_val_loss = boundary_record.get("val_loss")

    if final_val_loss is not None:
        final_val_loss = float(final_val_loss)
        if not math.isfinite(final_val_loss):
            raise ValueError(f"{source}: validation loss must be finite when present")
    return history_sec, history_tokens, final_val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer LM")
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, help="Resume model, optimizer, and iteration from a checkpoint")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=JSON_VALUE",
        help="Override a top-level config value, for example --set batch_size=64",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator or key not in config:
            raise ValueError(f"invalid config override: {override}")
        config[key] = json.loads(value)
    device = torch.device(args.device)
    torch.manual_seed(config.get("seed", 42))

    train_data = np.load(config["train_data"], mmap_mode="r")
    val_data = np.load(config["val_data"], mmap_mode="r")
    model = TransformerLM(
        config["vocab_size"], config["context_length"], config["d_model"], config["num_layers"],
        config["num_heads"], config["d_ff"], config.get("rope_theta", 10000.0),
        config.get("norm_style", "pre"), config.get("use_rope", True), config.get("ffn_variant", "swiglu"),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config["max_lr"], weight_decay=config["weight_decay"])
    start_iteration = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    if start_iteration > config["max_iters"]:
        raise ValueError(
            f"checkpoint iteration {start_iteration} exceeds configured max_iters {config['max_iters']}"
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.get("log_name", "metrics.jsonl")
    summary_path = output_dir / "summary.json"
    if args.resume:
        history_sec, history_tokens, final_val_loss = load_resume_metrics(
            log_path, summary_path, start_iteration,
        )
    else:
        if summary_path.exists() or (log_path.exists() and log_path.stat().st_size):
            raise FileExistsError(
                f"refusing to overwrite existing run history in {output_dir}; "
                "use --resume or choose a new output_dir"
            )
        log_path.write_text("", encoding="utf-8")
        history_sec = 0.0
        history_tokens = 0
        final_val_loss = None
    session_start = time.perf_counter()
    tokens_per_step = config["batch_size"] * config["context_length"]
    completed_steps = start_iteration
    status = "completed"

    for step in range(start_iteration + 1, config["max_iters"] + 1):
        schedule_iteration = step - 1
        lr = cosine_schedule(
            schedule_iteration, config["max_lr"], config["min_lr"],
            config["warmup_iters"], config["max_iters"] - 1,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        start = time.perf_counter()
        x, y = get_batch(train_data, config["batch_size"], config["context_length"], str(device))
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config, device):
            loss = cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            status = "non_finite_loss"
            record = {
                "step": step,
                "processed_tokens": history_tokens + (completed_steps - start_iteration) * tokens_per_step,
                "wall_clock_sec": history_sec + time.perf_counter() - session_start,
                "train_loss": None,
                "val_loss": final_val_loss,
                "lr": lr,
                "status": status,
            }
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(record, flush=True)
            break
        loss.backward()
        clip_gradients(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        completed_steps = step

        should_evaluate = (
            step == 1 or step % config["eval_interval"] == 0 or step == config["max_iters"]
        )
        if should_evaluate:
            final_val_loss = evaluate(model, val_data, config, str(device))
        if should_evaluate or step % config["checkpoint_interval"] == 0:
            record = {
                "step": step,
                "processed_tokens": history_tokens + (step - start_iteration) * tokens_per_step,
                "wall_clock_sec": history_sec + time.perf_counter() - session_start,
                "train_loss": loss.item(), "val_loss": final_val_loss, "lr": lr,
                "step_seconds": time.perf_counter() - start,
            }
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(record, flush=True)
        if step % config["checkpoint_interval"] == 0 or step == config["max_iters"]:
            save_checkpoint(model, optimizer, step, output_dir / f"checkpoint_{step}.pt")

    session_training_sec = time.perf_counter() - session_start
    session_processed_tokens = (completed_steps - start_iteration) * tokens_per_step
    total_training_sec = history_sec + session_training_sec
    total_processed_tokens = history_tokens + session_processed_tokens
    summary = {
        "status": status,
        "completed_steps": completed_steps,
        "requested_max_iters": config["max_iters"],
        "final_val_loss": final_val_loss,
        "total_training_sec": total_training_sec,
        "total_processed_tokens": total_processed_tokens,
        "average_tokens_per_sec": total_processed_tokens / total_training_sec,
        "resumed_from_iteration": start_iteration,
        "session_training_sec": session_training_sec,
        "session_processed_tokens": session_processed_tokens,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": {
            key: config[key] for key in (
                "vocab_size", "d_model", "num_layers", "num_heads", "d_ff", "context_length",
                "batch_size", "max_iters", "warmup_iters", "max_lr", "min_lr", "weight_decay",
                "max_grad_norm", "eval_interval", "eval_batches",
            )
        },
        "variant": {
            "norm_style": config.get("norm_style", "pre"),
            "use_rope": config.get("use_rope", True),
            "ffn_variant": config.get("ffn_variant", "swiglu"),
            "amp_dtype": config.get("amp_dtype"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
