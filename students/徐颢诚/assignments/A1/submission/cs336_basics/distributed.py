"""单机多 GPU DDP 训练的初始化与清理工具。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """当前训练进程在 DDP world 中的身份和应使用的设备。"""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed(device_name: str) -> DistributedContext:
    """从 torchrun 环境变量初始化 DDP；普通 python 启动时退化为单进程。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if device_name not in {"auto", "cuda"}:
            raise ValueError("DDP requires training.device to be 'auto' or 'cuda'")
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return DistributedContext(rank, local_rank, world_size, torch.device(f"cuda:{local_rank}"))

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    return DistributedContext(rank, local_rank, world_size, device)


def barrier(context: DistributedContext) -> None:
    """仅在 DDP 下同步全部 rank。"""
    if context.is_distributed:
        # NCCL 必须知道本 rank 使用哪张卡；否则在首次 barrier 时可能选择错误设备并卡住。
        dist.barrier(device_ids=[context.local_rank])


def cleanup_distributed(context: DistributedContext) -> None:
    """销毁 NCCL process group，释放分布式通信资源。"""
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
