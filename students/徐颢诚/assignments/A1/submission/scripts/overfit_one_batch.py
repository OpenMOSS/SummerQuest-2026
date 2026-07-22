"""固定一个小 batch 并反复训练，用于验证 Transformer 实现能否过拟合。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.logging import build_logger
from cs336_basics.nn import TransformerLM, cross_entropy
from cs336_basics.optim import AdamW, clip_grad_norm_
from scripts.train_lm import resolve_device


@hydra.main(version_base="1.3", config_path="../configs", config_name="overfit_one_batch")
def main(cfg: DictConfig) -> None:
    """采样一次 batch；之后每一步都用它更新，从而检查模型是否可过拟合。"""
    if cfg.training.steps <= 0 or cfg.training.batch_size <= 0 or cfg.training.log_interval <= 0:
        raise ValueError("training.steps, batch_size, and log_interval must be positive")

    run_directory = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, run_directory / "config.yaml", resolve=True)
    resolved_config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    logger = build_logger(cfg.logging.backends, run_directory, resolved_config)
    try:
        device = resolve_device(str(cfg.training.device))
        dtype = getattr(torch, str(cfg.model.dtype))
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)

        train_data = np.load(cfg.data.train_path, mmap_mode="r")
        if train_data.ndim != 1 or len(train_data) <= cfg.model.context_length:
            raise ValueError("train data must be a 1D array longer than context_length")
        if int(train_data.max()) >= cfg.data.vocab_size:
            raise ValueError("train data contains a token ID outside vocab_size")

        model = TransformerLM(
            cfg.data.vocab_size,
            cfg.model.context_length,
            cfg.model.d_model,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.d_ff,
            cfg.model.rope_theta,
            device=device,
            dtype=dtype,
        )
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )

        # 关键点：只采样一次，循环中绝不重新调用 get_batch。
        inputs, targets = get_batch(train_data, cfg.training.batch_size, cfg.model.context_length, device)
        started_at = time.perf_counter()
        final_loss = float("nan")
        for step in range(1, cfg.training.steps + 1):
            optimizer.zero_grad()
            loss = cross_entropy(model(inputs), targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_norm)
            optimizer.step()
            final_loss = loss.item()

            if step % cfg.training.log_interval == 0 or step == cfg.training.steps:
                metrics = {
                    "step": step,
                    "wall_clock_sec": round(time.perf_counter() - started_at, 3),
                    "train_loss": final_loss,
                    "lr": cfg.optimizer.lr,
                }
                logger.log_metrics(metrics)
                print(metrics, flush=True)

        checkpoint_path = run_directory / cfg.training.checkpoint_name
        save_checkpoint(model, optimizer, cfg.training.steps, checkpoint_path)
        logger.log_summary(
            {
                "run_name": cfg.run_name,
                "final_train_loss": final_loss,
                "total_training_time_sec": time.perf_counter() - started_at,
                "batch_size": cfg.training.batch_size,
                "context_length": cfg.model.context_length,
                "total_steps": cfg.training.steps,
                "checkpoint_path": str(checkpoint_path),
            }
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
