from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.nn import AdamW, TransformerLM, cross_entropy, gradient_clipping, lr_cosine_schedule
from scripts.plot_training_curves import plot_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer language model.")
    parser.add_argument("--train-data", required=True, help="Path to a 1D .npy array of token IDs.")
    parser.add_argument("--val-data", required=True, help="Path to a 1D .npy array of token IDs.")
    parser.add_argument("--out-dir", default=None, help="Exact directory for logs and checkpoints.")
    parser.add_argument("--run-root", default="runs", help="Parent directory used when --out-dir is omitted.")
    parser.add_argument("--experiment-name", default=None, help="Human-readable experiment name.")
    parser.add_argument("--no-auto-name", action="store_true", help="Disable timestamp/config suffix for auto out-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty fresh run directory.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-interrupted", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--norm-type", choices=["rmsnorm", "none"], default="rmsnorm")
    parser.add_argument("--norm-position", choices=["pre", "post"], default="pre")
    parser.add_argument("--pos-emb", choices=["rope", "none"], default="rope")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--status-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=float(os.environ.get("PEAK_FLOPS", 989e12)),
        help="Hardware peak FLOP/s used for MFU. Default is H100 BF16 dense peak, 989e12.",
    )
    return parser.parse_args()


def load_tokens(path: str | os.PathLike) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def slugify(text: str) -> str:
    chars = []
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", "."}:
            chars.append(char)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    name = args.experiment_name or "lm"
    if not args.no_auto_name:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        variant = []
        if args.d_ff != 1344:
            variant.append(f"dff{args.d_ff}")
        if args.norm_type != "rmsnorm":
            variant.append(f"norm{args.norm_type}")
        if args.norm_position != "pre":
            variant.append(f"normpos{args.norm_position}")
        if args.pos_emb != "rope":
            variant.append(f"posemb{args.pos_emb}")
        if args.ffn_type != "swiglu":
            variant.append(f"ffn{args.ffn_type}")
        variant_suffix = "-" + "-".join(variant) if variant else ""
        name = (
            f"{stamp}-{name}-vs{args.vocab_size}-ctx{args.context_length}-"
            f"dm{args.d_model}-l{args.num_layers}-h{args.num_heads}-bs{args.batch_size}-lr{args.max_lr:g}"
            f"{variant_suffix}"
        )
    return Path(args.run_root) / slugify(name)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_event(run_dir: Path, event: str, **fields) -> None:
    record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
    write_jsonl(run_dir / "events.jsonl", record)


def try_plot_curves(run_dir: Path) -> None:
    try:
        plot_path = plot_curves(run_dir)
    except Exception as exc:
        append_event(run_dir, "plot_failed", error=repr(exc))
        return
    if plot_path is not None:
        append_event(run_dir, "plot_written", path=str(plot_path))


def make_amp_context(use_amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def estimate_training_flops_per_token(model: torch.nn.Module, num_layers: int, d_model: int, context_length: int) -> int:
    n_params = sum(parameter.numel() for parameter in model.parameters())
    return 6 * n_params + 12 * num_layers * d_model * context_length


def calculate_mfu(tokens_per_sec: float | None, flops_per_token: int, peak_flops: float) -> float | None:
    if tokens_per_sec is None or peak_flops <= 0:
        return None
    return tokens_per_sec * flops_per_token / peak_flops


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        with make_amp_context(use_amp):
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = resolve_out_dir(args)
    if out_dir.exists() and any(out_dir.iterdir()) and args.resume is None and not args.overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite or choose another run directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    config_path = out_dir / "config.json"
    args_dict = vars(args).copy()
    args_dict["resolved_out_dir"] = str(out_dir)
    write_json(config_path, args_dict)
    append_event(out_dir, "run_started", out_dir=str(out_dir))

    train_data = load_tokens(args.train_data)
    val_data = load_tokens(args.val_data)
    if len(train_data) <= args.context_length or len(val_data) <= args.context_length:
        raise ValueError("train and val datasets must be longer than context_length")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_type=args.norm_type,
        norm_position=args.norm_position,
        pos_emb=args.pos_emb,
        ffn_type=args.ffn_type,
    ).to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    flops_per_token = estimate_training_flops_per_token(model, args.num_layers, args.d_model, args.context_length)
    model_params = sum(parameter.numel() for parameter in model.parameters())
    append_event(
        out_dir,
        "model_profile",
        model_params=model_params,
        flops_per_token=flops_per_token,
        peak_flops=args.peak_flops,
    )

    start_step = 0
    best_val_loss = math.inf
    best_step = None
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        move_optimizer_state_to_device(optimizer, args.device)
        summary_path = out_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best_val_loss = float(summary.get("best_val_loss", best_val_loss))
            best_step = summary.get("best_step", best_step)
        append_event(out_dir, "resumed", checkpoint=args.resume, start_step=start_step)

    use_amp = args.device.startswith("cuda") and args.dtype == "bfloat16"
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = start_step * args.batch_size * args.context_length
    model.train()

    iteration = start_step
    last_train_loss = None
    status = "running"
    write_json(out_dir / "status.json", {"status": status, "step": iteration})
    try:
        for step in range(start_step, args.steps):
            did_log = False
            did_eval = False
            did_checkpoint = False
            lr = lr_cosine_schedule(step, args.max_lr, args.min_lr, args.warmup_iters, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
            optimizer.zero_grad(set_to_none=True)
            with make_amp_context(use_amp):
                logits = model(x)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            gradient_clipping(model.parameters(), args.max_grad_norm)
            optimizer.step()

            iteration = step + 1
            last_train_loss = loss.item()
            tokens = iteration * args.batch_size * args.context_length
            if iteration % args.log_interval == 0 or iteration == 1:
                now = time.time()
                elapsed = now - start_time
                tokens_per_sec = tokens / elapsed if elapsed > 0 else None
                mfu = calculate_mfu(tokens_per_sec, flops_per_token, args.peak_flops)
                interval_elapsed = now - last_log_time
                interval_tokens = tokens - last_log_tokens
                interval_tokens_per_sec = interval_tokens / interval_elapsed if interval_elapsed > 0 else None
                interval_mfu = calculate_mfu(interval_tokens_per_sec, flops_per_token, args.peak_flops)
                record = {
                    "step": iteration,
                    "split": "train",
                    "loss": last_train_loss,
                    "ppl": math.exp(min(last_train_loss, 20.0)),
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "tokens": tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "mfu": mfu,
                    "mfu_percent": mfu * 100 if mfu is not None else None,
                    "interval_sec": interval_elapsed,
                    "interval_tokens": interval_tokens,
                    "interval_tokens_per_sec": interval_tokens_per_sec,
                    "interval_mfu": interval_mfu,
                    "interval_mfu_percent": interval_mfu * 100 if interval_mfu is not None else None,
                }
                write_jsonl(log_path, record)
                print(json.dumps(record, ensure_ascii=True), flush=True)
                last_log_time = now
                last_log_tokens = tokens
                did_log = True

            if iteration % args.eval_interval == 0 or iteration == args.steps:
                val_loss = estimate_loss(
                    model,
                    val_data,
                    args.batch_size,
                    args.context_length,
                    args.device,
                    args.eval_iters,
                    use_amp,
                )
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_step = iteration
                    if args.save_best:
                        save_checkpoint(model, optimizer, iteration, out_dir / "ckpt_best.pt")
                        append_event(out_dir, "best_checkpoint", step=iteration, val_loss=val_loss)
                record = {
                    "step": iteration,
                    "split": "val",
                    "loss": val_loss,
                    "ppl": math.exp(min(val_loss, 20.0)),
                    "lr": lr,
                    "elapsed_sec": time.time() - start_time,
                    "tokens": tokens,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "is_best": is_best,
                }
                write_jsonl(log_path, record)
                print(json.dumps(record, ensure_ascii=True), flush=True)
                did_eval = True

            if iteration % args.checkpoint_interval == 0 or iteration == args.steps:
                save_checkpoint(model, optimizer, iteration, out_dir / f"ckpt_{iteration:06d}.pt")
                save_checkpoint(model, optimizer, iteration, out_dir / "ckpt_latest.pt")
                append_event(out_dir, "checkpoint", step=iteration)
                did_checkpoint = True

            if (
                did_log
                or did_eval
                or did_checkpoint
                or iteration % args.status_interval == 0
                or iteration == args.steps
            ):
                write_json(
                    out_dir / "run_summary.json",
                    {
                        "status": "running",
                        "step": iteration,
                        "tokens": tokens,
                        "last_train_loss": last_train_loss,
                        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
                        "best_step": best_step,
                        "elapsed_sec": time.time() - start_time,
                        "out_dir": str(out_dir),
                    },
                )
                write_json(out_dir / "status.json", {"status": "running", "step": iteration})
        status = "completed"
        append_event(out_dir, "run_completed", step=iteration)
    except KeyboardInterrupt:
        status = "interrupted"
        if args.save_interrupted:
            save_checkpoint(model, optimizer, iteration, out_dir / "ckpt_interrupted.pt")
        append_event(out_dir, "run_interrupted", step=iteration)
        raise
    except Exception as exc:
        status = "failed"
        if args.save_interrupted:
            save_checkpoint(model, optimizer, iteration, out_dir / "ckpt_failed.pt")
        (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        append_event(out_dir, "run_failed", step=iteration, error=repr(exc))
        raise
    finally:
        summary = {
            "status": status,
            "step": iteration,
            "tokens": iteration * args.batch_size * args.context_length,
            "last_train_loss": last_train_loss,
            "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
            "best_step": best_step,
            "elapsed_sec": time.time() - start_time,
            "out_dir": str(out_dir),
        }
        write_json(out_dir / "run_summary.json", summary)
        write_json(out_dir / "status.json", {"status": status, "step": iteration})
        try_plot_curves(out_dir)


if __name__ == "__main__":
    main()
