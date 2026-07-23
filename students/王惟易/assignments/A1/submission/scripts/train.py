import time
import json
import torch
import argparse
from pathlib import Path
import numpy as np

from cs336_basics.training import train_step, get_batch, evaluate, save_checkpoint, load_checkpoint
from cs336_basics.optimizer import get_lr_cosine_schedule, AdamW
from cs336_basics.model import TransformerLM

def run_training(
        model,
        optimizer,
        train_data,
        val_data,
        *,
        device,
        batch_size,
        context_length,
        max_steps,
        max_learning_rate,
        min_learning_rate,
        warmup_iters,
        cosine_cycle_iters,
        max_l2_norm,
        log_interval,
        eval_interval,
        eval_batches,
        start_step=0,
        checkpoint_interval=None,
        checkpoint_path=None,
        wall_clock_offset=0.0,
        log_record=print,
):
    start_time = time.perf_counter()
    running_loss = torch.tensor(0.0, device=device)
    running_steps = 0

    for step in range(start_step, max_steps):
        # scheduler 必须在参数更新前设置
        lr = get_lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, batch_size, context_length, device)
        loss = train_step(model, optimizer, x, y, max_l2_norm=max_l2_norm)

        running_loss += loss.detach()
        running_steps += 1
        completed_steps = step + 1

        should_log = completed_steps % log_interval == 0
        should_eval = completed_steps % eval_interval == 0
        if completed_steps == max_steps:
            should_log = True
            should_eval = True

        if should_log or should_eval:
            if should_eval:
                val_loss = evaluate(model, val_data, batch_size, context_length, device, eval_batches)
            train_loss = (running_loss / running_steps).item()
            record = {
                "step": completed_steps,
                "wall_clock_sec": time.perf_counter() - start_time + wall_clock_offset,
                "train_loss": train_loss,
                "lr": lr,
            }
            if should_eval:
                record["val_loss"] = val_loss
            log_record(record)
            running_loss.zero_()
            running_steps = 0

        should_save_checkpoint = (
            checkpoint_path is not None and (
                completed_steps == max_steps or (
                    checkpoint_interval is not None and completed_steps % checkpoint_interval == 0
                )
            )
        )
        if should_save_checkpoint:
            save_checkpoint(model, optimizer, completed_steps, checkpoint_path)



class JsonlLogger:
    def __init__(self, path, append=False):
        self.path = path
        self.append = append
        self.file = None

    def _open(self):
        if self.file is None:
            mode = "a" if self.append else "w"
            self.file = open(self.path, mode, encoding="utf-8")

    def __call__(self, record):
        self._open()
        line = json.dumps(record, ensure_ascii=False)
        self.file.write(line + "\n")
        self.file.flush()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file is not None:
            self.file.close()
            self.file = None
        return False

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a decoder-only Transformer language model"
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON config file"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct the runtime and print summary without training"
    )

    return parser.parse_args(argv)

def load_config(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def resolve_path(config_path, configured_path):
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return config_path.parent / path

def read_wall_clock_offset(log_path):
    if not log_path.exists():
        return 0.0

    offset = 0.0
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                offset = float(record["wall_clock_sec"])

    return offset

def main():
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_path = resolve_path(config_path, config["data"]["train_path"])
    val_path = resolve_path(config_path, config["data"]["val_path"])
    train_data = np.memmap(train_path, dtype=config["data"]["dtype"], mode="r")
    val_data = np.memmap(val_path, dtype=config["data"]["dtype"], mode="r")

    model = TransformerLM(
        **config["model"],
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["max_learning_rate"],
        **config["optimizer"]
    )

    if args.dry_run:
        summary = {
            "device": str(device),
            "train_tokens": len(train_data),
            "val_tokens": len(val_data),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer": type(optimizer).__name__,
        }
        print(json.dumps(summary, indent=2))
        return

    log_path = resolve_path(config_path, config["output"]["log_path"])
    checkpoint_path = resolve_path(config_path, config["output"]["checkpoint_path"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    wall_clock_offset = 0.0

    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume.resolve(),
            model,
            optimizer,
        )
        wall_clock_offset = read_wall_clock_offset(log_path)

    with JsonlLogger(
        log_path,
        append=args.resume is not None,
    ) as logger:
        run_training(
            model,
            optimizer,
            train_data,
            val_data,
            device=device,
            context_length=config["model"]["context_length"],
            start_step=start_step,
            checkpoint_path=checkpoint_path,
            wall_clock_offset=wall_clock_offset,
            log_record=logger,
            **config["training"]
        )



if __name__ == "__main__":
    main()