from __future__ import annotations

import torch


def save_checkpoint(model, optimizer, iteration: int, out) -> None:
    # 三样一起存，少一个都没法老老实实接着跑。
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src, model, optimizer) -> int:
    save_data = torch.load(src, map_location="cpu")
    model.load_state_dict(save_data["model_state_dict"])
    optimizer.load_state_dict(save_data["optimizer_state_dict"])
    return int(save_data["iteration"])
