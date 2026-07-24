from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.cross_entropy import cross_entropy
from scripts.experiment_utils import (
    append_jsonl,
    apply_sets,
    clean_json,
    load_json,
    parse_dtype,
    project_path,
    select_device,
    sha256,
    write_json,
)
from scripts.train import build_model, load_tokens, resolve_config


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], int | None]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        iteration = payload.get("iteration")
    elif isinstance(payload, dict):
        state = payload
        iteration = None
    else:
        raise ValueError(f"unsupported checkpoint payload: {path}")
    if state and all(str(key).startswith("_orig_mod.") for key in state):
        state = {str(key).removeprefix("_orig_mod."): value for key, value in state.items()}
    return state, iteration


@torch.inference_mode()
def full_validation_loss(
    model: torch.nn.Module,
    token_ids: np.ndarray,
    context_length: int,
    batch_size: int,
    device: torch.device,
    max_tokens: int | None = None,
) -> tuple[float, int, int]:
    if len(token_ids) < 2:
        raise ValueError("validation token array must contain at least two tokens")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    cursor = 0
    limit = len(token_ids) - 1 if max_tokens is None else min(max_tokens, len(token_ids) - 1)

    while cursor < limit:
        seq_len = min(context_length, limit - cursor)
        if seq_len == context_length:
            starts = range(cursor, min(cursor + batch_size * context_length, limit), context_length)
            chunks = [np.asarray(token_ids[start : start + context_length + 1]) for start in starts]
            chunks = [chunk for chunk in chunks if len(chunk) == context_length + 1]
            cursor += len(chunks) * context_length
            if not chunks:
                continue
            batch = np.stack(chunks)
            x = torch.tensor(batch[:, :-1], dtype=torch.long, device=device)
            y = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
        else:
            chunk = np.asarray(token_ids[cursor : cursor + seq_len + 1])
            cursor += seq_len
            x = torch.tensor(chunk[:-1][None, :], dtype=torch.long, device=device)
            y = torch.tensor(chunk[1:][None, :], dtype=torch.long, device=device)

        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        tokens = int(y.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
        total_batches += 1

    return total_loss / total_tokens, total_tokens, total_batches


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute full validation loss for a saved CS336 checkpoint.")
    parser.add_argument("--config", required=True, type=Path, help="Training config used to build the model.")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to CHECKPOINT_DIR/final.pt, then best.pt.")
    parser.add_argument("--batch-size", type=int, help="Evaluation batch size; defaults to validation.batch_size.")
    parser.add_argument("--max-tokens", type=int, help="Debug/profiling cap. Omit for full validation.")
    parser.add_argument("--device", help="Device override: cpu, mps, cuda, or auto.")
    parser.add_argument("--output", type=Path, help="JSON output path.")
    parser.add_argument("--append-log", type=Path, help="Optional JSONL path for evaluation records.")
    parser.add_argument("--set", action="append", default=[], help="Override training config with dotted.path=JSON.")
    args = parser.parse_args()

    cfg = resolve_config(apply_sets(load_json(args.config), args.set))
    device = select_device(args.device or cfg["training"].get("device"))
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = ckpt_dir / "final.pt" if (ckpt_dir / "final.pt").is_file() else ckpt_dir / "best.pt"
    checkpoint = project_path(checkpoint).resolve(strict=True)
    batch_size = int(args.batch_size or cfg["validation"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    model = build_model(cfg, device, parse_dtype(cfg["training"].get("dtype")))
    state, iteration = checkpoint_state(checkpoint)
    model.load_state_dict(state, strict=True)
    val_data = load_tokens(cfg["data"]["val_dataset"], cfg["data"].get("dataset_dtype"))

    start = time.perf_counter()
    val_loss, evaluated_tokens, batches = full_validation_loss(
        model=model,
        token_ids=val_data,
        context_length=cfg["model"]["context_length"],
        batch_size=batch_size,
        device=device,
        max_tokens=args.max_tokens,
    )
    elapsed = time.perf_counter() - start
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_name": cfg["run"]["name"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_iteration": iteration,
        "val_loss": val_loss,
        "evaluated_tokens": evaluated_tokens,
        "validation_tokens_total": int(len(val_data)),
        "is_full_validation": args.max_tokens is None or evaluated_tokens >= int(len(val_data) - 1),
        "context_length": cfg["model"]["context_length"],
        "batch_size": batch_size,
        "batches": batches,
        "wall_clock_sec": elapsed,
        "device": str(device),
    }
    output = project_path(args.output or f"logs/{cfg['run']['name']}_full_val.json")
    write_json(output, record)
    if args.append_log:
        append_jsonl(args.append_log, record)
    print(clean_json(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
