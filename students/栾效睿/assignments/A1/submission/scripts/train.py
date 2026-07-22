from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cosine_lr import cosine_lr
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import data_loading
from cs336_basics.grad_clip import grad_clip
from cs336_basics.transformer_lm import TransformerLM
from scripts.experiment_utils import (
    append_jsonl,
    apply_sets,
    clean_json,
    load_json,
    parse_dtype,
    project_path,
    seed_all,
    select_device,
    set_dotted,
    write_json,
)


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name!r} must be an object")
    return dict(value)


def positive_int(value: Any, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def resolve_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    cfg = {k: section(raw, k) for k in ("run", "data", "model", "optimizer", "schedule", "training")}
    cfg["validation"] = section(raw, "validation")
    cfg["checkpoint"] = section(raw, "checkpoint")
    cfg["logging"] = section(raw, "logging")
    cfg["stability"] = section(raw, "stability")

    run_name = str(cfg["run"].get("name") or cfg["run"].get("experiment_name") or "run")
    cfg["run"]["name"] = run_name

    data = cfg["data"]
    for key in ("train_dataset", "val_dataset", "vocab_path", "merges_path"):
        data[key] = str(project_path(data[key]).resolve(strict=True))
    data["special_tokens"] = list(data.get("special_tokens", ["<|endoftext|>"]))

    model = cfg["model"]
    for key in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads"):
        model[key] = positive_int(model[key], f"model.{key}")
    if model["d_model"] % model["num_heads"]:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    model["d_ff"] = None if model.get("d_ff") is None else positive_int(model["d_ff"], "model.d_ff")
    model["theta"] = float(model.get("theta", 10000.0))
    model["eps"] = None if model.get("eps") is None else float(model["eps"])
    model["use_rope"] = bool(model.get("use_rope", True))
    model["norm_mode"] = str(model.get("norm_mode", "pre"))
    model["ffn_type"] = str(model.get("ffn_type", "swiglu"))
    if model["norm_mode"] not in {"pre", "post", "none"}:
        raise ValueError("model.norm_mode must be pre, post, or none")
    if model["ffn_type"] not in {"swiglu", "silu"}:
        raise ValueError("model.ffn_type must be swiglu or silu")

    training = cfg["training"]
    training["batch_size"] = positive_int(training["batch_size"], "training.batch_size")
    training["seed"] = None if training.get("seed") is None else int(training["seed"])
    training["dtype"] = training.get("dtype")
    tokens_per_step = training["batch_size"] * model["context_length"]
    if training.get("target_tokens") is None:
        steps = positive_int(training["num_steps"], "training.num_steps")
        training["target_tokens"] = steps * tokens_per_step
    else:
        training["target_tokens"] = positive_int(training["target_tokens"], "training.target_tokens")
        steps = math.ceil(training["target_tokens"] / tokens_per_step)
    training["num_steps"] = steps
    training["tokens_per_step"] = tokens_per_step
    training["log_every"] = positive_int(training.get("log_every", 10), "training.log_every")
    training["print_every"] = positive_int(training.get("print_every", 10), "training.print_every")
    training["torch_compile"] = bool(training.get("torch_compile", False))
    training["compile_mode"] = training.get("compile_mode")
    training["autocast_dtype"] = training.get("autocast_dtype")
    training["max_wall_clock_sec"] = training.get("max_wall_clock_sec")
    if training["max_wall_clock_sec"] is not None:
        training["max_wall_clock_sec"] = float(training["max_wall_clock_sec"])
        if training["max_wall_clock_sec"] <= 0:
            raise ValueError("training.max_wall_clock_sec must be positive")

    schedule = cfg["schedule"]
    schedule["lr_max"] = float(schedule["lr_max"])
    schedule["lr_min"] = float(schedule.get("lr_min", schedule["lr_max"] * float(schedule.get("lr_min_ratio", 0.1))))
    warmup = schedule.get("warmup_steps")
    if warmup is None:
        warmup = round(steps * float(schedule.get("warmup_fraction", 0.025)))
    schedule["warmup_steps"] = max(0, min(int(warmup), steps - 1))
    schedule["cycle_steps"] = int(schedule.get("cycle_steps", steps))
    if not 0 <= schedule["lr_min"] < schedule["lr_max"]:
        raise ValueError("schedule must satisfy 0 <= lr_min < lr_max")
    if schedule["cycle_steps"] <= schedule["warmup_steps"]:
        raise ValueError("schedule.cycle_steps must be greater than schedule.warmup_steps")

    validation = cfg["validation"]
    validation["batch_size"] = positive_int(validation.get("batch_size", training["batch_size"]), "validation.batch_size")
    validation["every_steps"] = validation.get("every_steps")
    validation["every_tokens"] = validation.get("every_tokens")
    validation["max_tokens"] = validation.get("max_tokens")
    if validation["every_steps"] is not None:
        validation["every_steps"] = positive_int(validation["every_steps"], "validation.every_steps")
    if validation["every_tokens"] is not None:
        validation["every_tokens"] = positive_int(validation["every_tokens"], "validation.every_tokens")
    if validation["max_tokens"] is not None:
        validation["max_tokens"] = positive_int(validation["max_tokens"], "validation.max_tokens")

    checkpoint = cfg["checkpoint"]
    checkpoint["enabled"] = bool(checkpoint.get("enabled", True))
    checkpoint["dir"] = str(project_path(checkpoint.get("dir", f"checkpoint/{run_name}")))
    checkpoint["save_every"] = checkpoint.get("save_every")
    checkpoint["save_best"] = bool(checkpoint.get("save_best", True))
    if checkpoint["save_every"] is not None:
        checkpoint["save_every"] = positive_int(checkpoint["save_every"], "checkpoint.save_every")

    logging = cfg["logging"]
    logging["path"] = str(project_path(logging.get("path", f"logs/{run_name}.jsonl")))
    logging["summary_path"] = str(project_path(logging.get("summary_path", f"logs/{run_name}.summary.json")))
    cfg["optimizer"].setdefault("betas", [0.9, 0.999])
    return cfg


def load_tokens(path: str, dtype: str | None) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        return np.load(path_obj, mmap_mode="r")
    return np.memmap(path_obj, dtype=np.dtype(dtype or "int64"), mode="r")


def build_model(cfg: Mapping[str, Any], device: torch.device, dtype: torch.dtype | None) -> TransformerLM:
    m = cfg["model"]
    return TransformerLM(
        d_model=m["d_model"],
        num_heads=m["num_heads"],
        vocab_size=m["vocab_size"],
        num_layers=m["num_layers"],
        max_seq_len=m["context_length"],
        d_ff=m["d_ff"],
        theta=m["theta"],
        use_rope=m["use_rope"],
        eps=m["eps"],
        device=device,
        dtype=dtype,
        norm_mode=m["norm_mode"],
        ffn_type=m["ffn_type"],
    )


def lm_loss(model: TransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))


def maybe_autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def autocast_dtype(cfg: Mapping[str, Any], device: torch.device) -> torch.dtype | None:
    dtype = parse_dtype(cfg["training"].get("autocast_dtype"))
    if dtype is None:
        return None
    if device.type != "cuda":
        return None
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("training.autocast_dtype=bfloat16 requires CUDA bf16 support")
    return dtype


def maybe_compile(model: TransformerLM, cfg: Mapping[str, Any], device: torch.device) -> tuple[TransformerLM, Literal[
    False]] | tuple[Callable[[Any], Any] | Any, Literal[True]]:
    if not cfg["training"].get("torch_compile"):
        return model, False
    if device.type != "cuda":
        print(f"torch.compile requested but skipped on device={device}", flush=True)
        return model, False
    mode = cfg["training"].get("compile_mode")
    if mode in {None, "default"}:
        return torch.compile(model), True
    return torch.compile(model, mode=str(mode)), True


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    data: np.ndarray,
    cfg: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    context = cfg["model"]["context_length"]
    max_tokens = min(int(cfg["validation"]["max_tokens"] or len(data)), len(data) - context - 1)
    starts = list(range(0, max_tokens, context))
    losses: list[float] = []
    for i in range(0, len(starts), cfg["validation"]["batch_size"]):
        chunks = [np.asarray(data[s : s + context + 1]) for s in starts[i : i + cfg["validation"]["batch_size"]]]
        if not chunks or any(len(chunk) != context + 1 for chunk in chunks):
            continue
        batch = np.stack(chunks)
        x = torch.tensor(batch[:, :-1], dtype=torch.long, device=device)
        y = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
        with maybe_autocast(device, amp_dtype):
            losses.append(float(lm_loss(model, x, y).item()))
    model.train()
    if not losses:
        raise ValueError("validation produced no batches; check validation.max_tokens and dataset length")
    return sum(losses) / len(losses)


def metric_record(step: int, elapsed: float, tokens: int, train_loss: float, val_loss: float | None, lr: float) -> dict[str, Any]:
    record = {
        "step": step,
        "wall_clock_sec": elapsed,
        "processed_tokens": tokens,
        "train_loss": train_loss,
        "lr": lr,
    }
    if val_loss is not None:
        record["val_loss"] = val_loss
    return record


def train(cfg: dict[str, Any], resume: str | None = None, reset_log: bool = True) -> dict[str, Any]:
    device = select_device(cfg["training"].get("device"))
    if device.type == "cuda" and cfg["training"].get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg["training"]["float32_matmul_precision"])
    seed_all(cfg["training"]["seed"])

    train_data = load_tokens(cfg["data"]["train_dataset"], cfg["data"].get("dataset_dtype"))
    val_data = load_tokens(cfg["data"]["val_dataset"], cfg["data"].get("dataset_dtype"))
    raw_model = build_model(cfg, device, parse_dtype(cfg["training"].get("dtype")))
    amp_dtype = autocast_dtype(cfg, device)
    opt = AdamW(
        raw_model.parameters(),
        lr=cfg["schedule"]["lr_max"],
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
        betas=tuple(float(x) for x in cfg["optimizer"]["betas"]),
        eps=float(cfg["optimizer"].get("eps", 1e-8)),
    )
    start_step = load_checkpoint(project_path(resume), raw_model, opt) if resume else 0
    model, compile_enabled = maybe_compile(raw_model, cfg, device)

    log_path = Path(cfg["logging"]["path"])
    if reset_log and start_step == 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    if cfg["checkpoint"]["enabled"]:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        write_json(ckpt_dir / "config.json", cfg)

    best_val = math.inf
    last_val: float | None = None
    last_train = math.inf
    status = "completed"
    stop_reason = "max_steps"
    next_eval_tokens = cfg["validation"].get("every_tokens")
    completed_steps = start_step
    start_time = time.perf_counter()

    for step in range(start_step + 1, cfg["training"]["num_steps"] + 1):
        if cfg["training"]["max_wall_clock_sec"] is not None and time.perf_counter() - start_time >= cfg["training"]["max_wall_clock_sec"]:
            status = "time_limit"
            stop_reason = "max_wall_clock_sec"
            break

        lr = cosine_lr(step, cfg["schedule"]["lr_max"], cfg["schedule"]["lr_min"], cfg["schedule"]["warmup_steps"], cfg["schedule"]["cycle_steps"])
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = data_loading(train_data, cfg["training"]["batch_size"], cfg["model"]["context_length"], device=device)
        opt.zero_grad(set_to_none=True)
        with maybe_autocast(device, amp_dtype):
            loss = lm_loss(model, x, y)
        last_train = float(loss.detach().item())
        if not math.isfinite(last_train) or last_train > float(cfg["stability"].get("divergence_loss_threshold", math.inf)):
            status = "diverged"
            stop_reason = "nonfinite_or_large_train_loss"
            elapsed = time.perf_counter() - start_time
            record = metric_record(step, elapsed, (step - 1) * cfg["training"]["tokens_per_step"], last_train, last_val, lr)
            record.update({"status": status, "stop_reason": stop_reason})
            append_jsonl(log_path, record)
            break
        loss.backward()
        if cfg["optimizer"].get("grad_clip_max_norm") is not None:
            grad_clip(
                raw_model.parameters(),
                float(cfg["optimizer"]["grad_clip_max_norm"]),
                float(cfg["optimizer"].get("grad_clip_eps", 1e-6)),
            )
        opt.step()
        completed_steps = step

        processed = step * cfg["training"]["tokens_per_step"]
        eval_by_step = cfg["validation"]["every_steps"] and step % cfg["validation"]["every_steps"] == 0
        eval_by_token = next_eval_tokens and processed >= next_eval_tokens
        should_eval = step == 1 or step == cfg["training"]["num_steps"] or eval_by_step or eval_by_token
        if eval_by_token:
            while next_eval_tokens and processed >= next_eval_tokens:
                next_eval_tokens += cfg["validation"]["every_tokens"]
        if should_eval:
            last_val = validate(model, val_data, cfg, device, amp_dtype)
            if last_val < best_val:
                best_val = last_val
                if cfg["checkpoint"]["enabled"] and cfg["checkpoint"]["save_best"]:
                    save_checkpoint(raw_model, opt, step, ckpt_dir / "best.pt")

        elapsed = time.perf_counter() - start_time
        time_limit_reached = (
            cfg["training"]["max_wall_clock_sec"] is not None
            and elapsed >= cfg["training"]["max_wall_clock_sec"]
        )
        if time_limit_reached:
            status = "time_limit"
            stop_reason = "max_wall_clock_sec"

        logged_val = last_val if should_eval else None
        should_log = should_eval or step % cfg["training"]["log_every"] == 0
        if should_log or time_limit_reached:
            record = metric_record(step, elapsed, processed, last_train, logged_val, lr)
            if time_limit_reached:
                record.update({"status": status, "stop_reason": stop_reason})
            append_jsonl(log_path, record)
        if cfg["checkpoint"]["enabled"] and cfg["checkpoint"]["save_every"] and step % cfg["checkpoint"]["save_every"] == 0:
            save_checkpoint(raw_model, opt, step, ckpt_dir / f"step_{step}.pt")
        if step % cfg["training"]["print_every"] == 0 or should_eval:
            message = f"{cfg['run']['name']} step={step} train={last_train:.4f}"
            if should_eval:
                message += f" val={last_val}"
            message += f" lr={lr:.3g} time={elapsed:.1f}s"
            if time_limit_reached:
                message += " status=time_limit"
            print(message, flush=True)
        if time_limit_reached:
            break

    final_step = completed_steps
    if cfg["checkpoint"]["enabled"] and status in {"completed", "time_limit"}:
        save_checkpoint(raw_model, opt, final_step, ckpt_dir / "final.pt")
    summary = {
        "run_name": cfg["run"]["name"],
        "status": status,
        "stop_reason": stop_reason,
        "final_step": final_step,
        "processed_tokens": final_step * cfg["training"]["tokens_per_step"],
        "final_train_loss": last_train,
        "final_val_loss": last_val,
        "best_val_loss": None if best_val == math.inf else best_val,
        "wall_clock_sec": time.perf_counter() - start_time,
        "log_path": cfg["logging"]["path"],
        "checkpoint_dir": cfg["checkpoint"]["dir"] if cfg["checkpoint"]["enabled"] else None,
        "key_config": {
            "batch_size": cfg["training"]["batch_size"],
            "context_length": cfg["model"]["context_length"],
            "d_model": cfg["model"]["d_model"],
            "d_ff": cfg["model"]["d_ff"],
            "num_layers": cfg["model"]["num_layers"],
            "num_heads": cfg["model"]["num_heads"],
            "vocab_size": cfg["model"]["vocab_size"],
            "lr_max": cfg["schedule"]["lr_max"],
            "max_wall_clock_sec": cfg["training"]["max_wall_clock_sec"],
            "torch_compile": cfg["training"]["torch_compile"],
            "torch_compile_enabled": compile_enabled,
            "autocast_dtype": cfg["training"]["autocast_dtype"],
            "norm_mode": cfg["model"]["norm_mode"],
            "use_rope": cfg["model"]["use_rope"],
            "ffn_type": cfg["model"]["ffn_type"],
        },
    }
    write_json(cfg["logging"]["summary_path"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer LM from a JSON config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted.path=JSON.")
    parser.add_argument("--run-name", help="Override run.name.")
    parser.add_argument("--resume", help="Resume from a checkpoint created by cs336_basics.checkpointing.")
    parser.add_argument("--append-log", action="store_true", help="Append to the configured log file.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the config without training.")
    args = parser.parse_args()

    raw = apply_sets(load_json(args.config), args.set)
    if args.run_name:
        set_dotted(raw, "run.name", args.run_name)
    cfg = resolve_config(raw)
    if args.dry_run:
        print(clean_json(cfg))
        return 0
    summary = train(cfg, resume=args.resume, reset_log=not args.append_log)
    print(clean_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
