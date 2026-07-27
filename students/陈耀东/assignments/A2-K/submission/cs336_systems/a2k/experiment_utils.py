"""A2 正式实验共用的测量工具。

这个模块把不同 benchmark 中容易被分别实现、随后逐渐产生口径差异的部分
集中起来：

* 延迟样本的均值、标准差、变异系数和 p20/p50/p80；
* RTX 4090 正式实验要求的 23 GiB PyTorch allocator 上限；
* CUDA 显存四个常用指标和硬件元数据。

函数不负责创建模型或输入，因此可以在第一次 CUDA 大显存分配之前调用
``configure_cuda_allocator``，也可以被 CPU 测试直接验证。
"""

from __future__ import annotations

import platform
import statistics
import sys
from typing import Any

import torch


MIB = 1024**2
GIB = 1024**3


def percentile(values: list[float], quantile: float) -> float:
    """用线性插值计算分位数，避免不同环境的 numpy 版本造成口径差异。

    ``quantile`` 取值范围是 ``[0, 1]``。这里使用与常见统计工具一致的
    ``(n - 1) * q`` 位置定义；例如 p50 在奇数样本时就是中位数。
    """

    if not values:
        raise ValueError("at least one sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(values: list[float]) -> dict[str, Any]:
    """返回正式实验表所需的统计量，并保留轻量原始样本。

    变异系数 ``cv`` 是标准差除以均值。它比单独看标准差更适合比较不同
    shape 的稳定性，但均值为零时没有定义，因此返回 ``None``。
    """

    if not values:
        raise ValueError("at least one latency sample is required")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples_ms": [float(value) for value in values],
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean if mean else None,
        "p20_ms": percentile(values, 0.20),
        "p50_ms": percentile(values, 0.50),
        "p80_ms": percentile(values, 0.80),
        "min_ms": min(values),
        "max_ms": max(values),
        "measurement_count": len(values),
    }


def configure_cuda_allocator(device: int | torch.device = 0, limit_gib: float | None = None) -> dict[str, Any] | None:
    """在第一次大显存分配之前设置 PyTorch allocator 上限。

    A2-K 的正式环境要求把单进程 PyTorch allocator 限制在 23 GiB。4090
    的物理总显存通常显示为约 24 GiB，因此不能只检查设备名称；这里根据
    实际总显存计算 fraction，并把完整配置写入结果，便于之后审计。

    这个函数只设置 PyTorch 进程的上限，不会改变 CUDA 驱动报告的物理显存。
    ``limit_gib=None`` 时不调用 setter，适合旧版不要求显存上限的实验。
    """

    if limit_gib is None:
        return None
    if limit_gib <= 0:
        raise ValueError("allocator limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("allocator limits require CUDA")

    if isinstance(device, int):
        device_index = device
    else:
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    total_bytes = int(properties.total_memory)
    limit_bytes = int(limit_gib * GIB)
    fraction = min(1.0, limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    return {
        "allocator_limit_gib": limit_gib,
        "allocator_limit_bytes": limit_bytes,
        "allocator_fraction": fraction,
        "gpu_total_memory_mib": total_bytes / MIB,
        "allocator_enforced": True,
    }


def cuda_memory_stats(device: torch.device | str = "cuda") -> dict[str, float] | None:
    """统一记录 allocated、reserved 以及两种峰值显存，单位为 MiB。"""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return {
        "allocated_mib": torch.cuda.memory_allocated(resolved) / MIB,
        "reserved_mib": torch.cuda.memory_reserved(resolved) / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(resolved) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(resolved) / MIB,
    }


def hardware_metadata(device: torch.device | str = "cuda") -> dict[str, Any]:
    """返回不含账号、路径和进程信息的公开实验环境摘要。"""

    resolved = torch.device(device)
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_mib": properties.total_memory / MIB,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    else:
        metadata["gpu_name"] = None
    return metadata
