from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.nn import cross_entropy
from cs336_basics.optim import AdamW, clip_grad_norm_, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.transformer import TransformerLM

from scripts.utils import Timer, append_jsonl, get_device, load_json, save_json, set_seed, tqdm_disabled


def _eval_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in tqdm(
            range(num_batches),
            desc="eval",
            leave=False,
            disable=tqdm_disabled(),
        ):
            x, y = get_batch(dataset, batch_size, context_length, str(device))
            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_json(args.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = get_device(args.device)

    train_tokens_path = cfg["train_tokens_path"]
    val_tokens_path = cfg["val_tokens_path"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_train = np.load(train_tokens_path, mmap_mode="r")
    dataset_val = np.load(val_tokens_path, mmap_mode="r")

    model_cfg = cfg["model"]
    vocab_size = int(model_cfg["vocab_size"])
    context_length = int(model_cfg["context_length"])
    d_model = int(model_cfg["d_model"])
    num_layers = int(model_cfg["num_layers"])
    num_heads = int(model_cfg["num_heads"])
    d_ff = int(model_cfg["d_ff"])
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    use_rope = bool(model_cfg.get("use_rope", True))
    use_rmsnorm = bool(model_cfg.get("use_rmsnorm", True))
    prenorm = bool(model_cfg.get("prenorm", True))
    ffn_type = str(model_cfg.get("ffn_type", "swiglu"))
    eps = float(model_cfg.get("eps", 1e-5))

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        use_rope=use_rope,
        use_rmsnorm=use_rmsnorm,
        prenorm=prenorm,
        ffn_type=ffn_type,
        eps=eps,
    ).to(device)

    optim_cfg = cfg["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(optim_cfg["learning_rate"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        eps=float(optim_cfg.get("eps", 1e-8)),
    )

    iter0 = 0
    if args.resume is not None:
        iter0 = load_checkpoint(args.resume, model, optimizer)

    save_json(out_dir / "model_config.json", model_cfg)
    save_json(out_dir / "train_config.json", cfg)

    batch_size = int(cfg["batch_size"])
    max_steps = int(cfg["max_steps"])
    eval_interval = int(cfg.get("eval_interval", 500))
    save_interval = int(cfg.get("save_interval", 1000))
    eval_batches = int(cfg.get("eval_batches", 10))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    sched_cfg = cfg["schedule"]
    max_lr = float(sched_cfg["max_learning_rate"])
    min_lr = float(sched_cfg["min_learning_rate"])
    warmup_iters = int(sched_cfg["warmup_iters"])
    cosine_cycle_iters = int(sched_cfg["cosine_cycle_iters"])

    log_path = out_dir / "train.jsonl"
    timer = Timer.start_now()

    for it in tqdm(
        range(iter0, max_steps),
        desc="train",
        initial=iter0,
        total=max_steps,
        disable=tqdm_disabled(),
    ):
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(dataset_train, batch_size, context_length, str(device))
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        row = {"step": it + 1, "wall_clock_sec": timer.elapsed(), "train_loss": float(loss.detach().cpu()), "lr": lr}

        if (it + 1) % eval_interval == 0:
            row["val_loss"] = _eval_loss(
                model=model,
                dataset=dataset_val,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                num_batches=eval_batches,
            )

        append_jsonl(log_path, row)
        if not tqdm_disabled():
            tqdm.write(
                f"step={row['step']} train_loss={row['train_loss']:.4f} lr={row['lr']:.3e}"
                + (f" val_loss={row['val_loss']:.4f}" if "val_loss" in row else "")
            )

        if (it + 1) % save_interval == 0:
            ckpt_path = out_dir / f"checkpoint_{it+1}.pt"
            save_checkpoint(model, optimizer, it + 1, ckpt_path)

    final_ckpt_path = out_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, max_steps, final_ckpt_path)


if __name__ == "__main__":
    main()
