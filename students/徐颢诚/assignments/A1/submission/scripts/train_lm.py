"""使用 Hydra 配置训练 Transformer LM。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from omegaconf import DictConfig, OmegaConf

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.distributed import barrier, cleanup_distributed, setup_distributed
from cs336_basics.logging import build_logger
from cs336_basics.nn import TransformerLM, cross_entropy
from cs336_basics.optim import AdamW, clip_grad_norm_, get_lr_cosine_schedule


def resolve_device(device_name: str) -> torch.device:
    """为单进程调用者解析设备；主训练入口使用 setup_distributed。"""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    validation_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int,
) -> float:
    """随机采样多个验证 batch，返回平均交叉熵。"""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(validation_data, batch_size, context_length, device)
        losses.append(cross_entropy(model(inputs), targets).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def validate_config(cfg: DictConfig) -> int:
    """在开始大规模训练前尽早发现不合法配置。"""
    if cfg.training.steps <= 0 or cfg.training.eval_batches <= 0:
        raise ValueError("training.steps and training.eval_batches must be positive")
    if min(cfg.training.log_interval, cfg.training.eval_interval, cfg.training.checkpoint_interval) <= 0:
        raise ValueError("log/eval/checkpoint intervals must be positive")
    cycle_steps = cfg.scheduler.cosine_cycle_steps or cfg.training.steps
    if not 0 <= cfg.scheduler.warmup_steps < cycle_steps:
        raise ValueError("scheduler.warmup_steps must satisfy 0 <= warmup_steps < cosine_cycle_steps")
    return int(cycle_steps)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_lm")
def main(cfg: DictConfig) -> None:
    """运行训练、验证、checkpoint 和实验日志流程。"""
    cosine_cycle_steps = validate_config(cfg)
    distributed = setup_distributed(str(cfg.training.device))
    run_directory = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger = None
    if distributed.is_main_process:
        # Hydra 会保存 .hydra/config.yaml；这里额外将最终解析后的配置放在 run 根目录。
        OmegaConf.save(cfg, run_directory / "config.yaml", resolve=True)
        resolved_config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
        logger = build_logger(cfg.logging.backends, run_directory, resolved_config)
    barrier(distributed)

    try:
        device = distributed.device
        dtype = getattr(torch, str(cfg.model.dtype))
        # 各 rank 采样不同 batch；DDP 会在构造时将 rank 0 的参数广播到所有 GPU。
        torch.manual_seed(cfg.training.seed + distributed.rank)
        np.random.seed(cfg.training.seed + distributed.rank)

        # mmap_mode='r' 让 token 文件按需读取，而非一次性占用数 GB 内存。
        train_data = np.load(cfg.data.train_path, mmap_mode="r")
        valid_data = np.load(cfg.data.valid_path, mmap_mode="r")
        for name, dataset in (("train", train_data), ("valid", valid_data)):
            if dataset.ndim != 1 or len(dataset) <= cfg.model.context_length:
                raise ValueError(f"{name} data must be a 1D array longer than context_length")
            if int(dataset.max()) >= cfg.data.vocab_size:
                raise ValueError(f"{name} data contains a token ID outside vocab_size")

        raw_model = TransformerLM(
            cfg.data.vocab_size,
            cfg.model.context_length,
            cfg.model.d_model,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.d_ff,
            cfg.model.rope_theta,
            device=device,
            dtype=dtype,
            use_rmsnorm=bool(cfg.model.get("use_rmsnorm", True)),
            use_rope=bool(cfg.model.get("use_rope", True)),
            use_swiglu=bool(cfg.model.get("use_swiglu", True)),
            norm_position=str(cfg.model.get("norm_position", "pre")),
        )
        optimizer = AdamW(
            raw_model.parameters(),
            lr=cfg.optimizer.max_lr,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )
        start_step = load_checkpoint(cfg.training.resume, raw_model, optimizer) if cfg.training.resume else 0
        if not 0 <= start_step <= cfg.training.steps:
            raise ValueError("checkpoint iteration must be between 0 and training.steps")
        model: torch.nn.Module
        if distributed.is_distributed:
            model = DistributedDataParallel(raw_model, device_ids=[distributed.local_rank], output_device=distributed.local_rank)
        else:
            model = raw_model

        checkpoint_path = run_directory / cfg.training.checkpoint_name
        started_at = time.perf_counter()
        last_validation_loss: float | None = None
        # 以全局 batch 定义 token 数，因而单卡和 DDP run 的日志可直接比较。
        tokens_per_update = cfg.training.batch_size * distributed.world_size * cfg.model.context_length
        for step in range(start_step, cfg.training.steps):
            restart_on_resume = bool(cfg.scheduler.get("restart_on_resume", False))
            schedule_step = step - start_step if restart_on_resume and start_step > 0 else step
            learning_rate = get_lr_cosine_schedule(
                schedule_step,
                cfg.optimizer.max_lr,
                cfg.optimizer.min_lr,
                cfg.scheduler.warmup_steps,
                cosine_cycle_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad()
            inputs, targets = get_batch(train_data, cfg.training.batch_size, cfg.model.context_length, device)
            loss = cross_entropy(model(inputs), targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_norm)
            optimizer.step()

            completed_step = step + 1
            should_evaluate = completed_step % cfg.training.eval_interval == 0 or completed_step == cfg.training.steps
            should_log = completed_step % cfg.training.log_interval == 0 or should_evaluate
            if should_evaluate:
                if distributed.is_main_process:
                    last_validation_loss = evaluate(
                        # 验证只在 rank 0 做；绕过 DDP wrapper，避免它发起其他 rank 不参与的 buffer 同步。
                        raw_model,
                        valid_data,
                        cfg.training.batch_size,
                        cfg.model.context_length,
                        device,
                        cfg.training.eval_batches,
                    )
                barrier(distributed)
            if should_log and distributed.is_main_process:
                metrics: dict[str, float | int] = {
                    "step": completed_step,
                    "processed_tokens": completed_step * tokens_per_update,
                    "wall_clock_sec": round(time.perf_counter() - started_at, 3),
                    "train_loss": loss.item(),
                    "lr": learning_rate,
                }
                if should_evaluate and last_validation_loss is not None:
                    metrics["val_loss"] = last_validation_loss
                assert logger is not None
                logger.log_metrics(metrics)
                print(metrics, flush=True)
            if completed_step % cfg.training.checkpoint_interval == 0 and distributed.is_main_process:
                save_checkpoint(raw_model, optimizer, completed_step, checkpoint_path)
            if completed_step % cfg.training.checkpoint_interval == 0:
                barrier(distributed)

        if distributed.is_main_process:
            save_checkpoint(raw_model, optimizer, cfg.training.steps, checkpoint_path)
        barrier(distributed)
        if last_validation_loss is None and distributed.is_main_process:
            last_validation_loss = evaluate(
                raw_model,
                valid_data,
                cfg.training.batch_size,
                cfg.model.context_length,
                device,
                cfg.training.eval_batches,
            )
        if distributed.is_main_process:
            summary = {
                "run_name": cfg.run_name,
                "final_val_loss": last_validation_loss,
                "total_training_time_sec": time.perf_counter() - started_at,
                "d_model": cfg.model.d_model,
                "num_layers": cfg.model.num_layers,
                "num_heads": cfg.model.num_heads,
                "context_length": cfg.model.context_length,
                "use_rmsnorm": bool(cfg.model.get("use_rmsnorm", True)),
                "use_rope": bool(cfg.model.get("use_rope", True)),
                "use_swiglu": bool(cfg.model.get("use_swiglu", True)),
                "norm_position": str(cfg.model.get("norm_position", "pre")),
                "batch_size_per_gpu": cfg.training.batch_size,
                "global_batch_size": cfg.training.batch_size * distributed.world_size,
                "world_size": distributed.world_size,
                "total_steps": cfg.training.steps,
                "processed_tokens": cfg.training.steps * tokens_per_update,
                "device": str(device),
                "checkpoint_path": str(checkpoint_path),
            }
            assert logger is not None
            logger.log_summary(summary)
            print(OmegaConf.to_yaml(OmegaConf.create(summary)), flush=True)
    finally:
        if logger is not None:
            logger.close()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
