"""A2 Single-GPU Memory：saved tensors 与 activation checkpointing 实验。

本文件对应讲义第三部分的两个观察对象：

1. ``saved-tensors``：使用 ``torch.autograd.graph.saved_tensors_hooks`` 记录
   RMSNorm 或 TransformerBlock 为 backward 保存了哪些张量；
2. ``checkpoint``：将 N 个 TransformerBlock 按固定组大小包进非嵌套
   ``torch.utils.checkpoint``，测量重计算粒度对峰值显存和时间的影响。

这里既记录 allocator 的绝对峰值，也记录相对 forward 前基线的增量。绝对值
回答“这张卡能否运行”，增量更接近“这一策略额外保存了多少训练中间状态”。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import RMSNorm, RotaryEmbedding, TransformerBlock
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k.experiment_utils import configure_cuda_allocator, hardware_metadata, summarize_samples


MIB = 1024**2


def synchronize(device: torch.device) -> None:
    """CUDA 计时前后等待设备完成，CPU 无需同步。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(device: torch.device, function):
    """执行函数并返回结果与同步墙钟时间。"""

    synchronize(device)
    start = time.perf_counter_ns()
    result = function()
    synchronize(device)
    return result, (time.perf_counter_ns() - start) / 1_000_000


def autocast_context(device: torch.device, precision: str):
    """让正式 checkpoint 实验使用 BF16 计算，同时保留 FP32 参数。

    ``AdamW`` 仍然接收 FP32 参数和 FP32 梯度；autocast 只改变适合低精度
    的 forward 算子。这正是 A2-K 要求观察的“参数精度”和“计算精度”分离。
    """

    if precision == "fp32":
        return nullcontext()
    if precision == "bf16-mixed" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("precision must be fp32 or bf16-mixed on CUDA")


def cuda_memory_mib(device: torch.device) -> dict[str, float] | None:
    """读取当前与峰值 allocator 指标。"""

    if device.type != "cuda":
        return None
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / MIB,
        "reserved_mib": torch.cuda.memory_reserved(device) / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
    }


def storage_pointer(tensor: torch.Tensor) -> int:
    """返回底层 storage 地址；不同 view 共享同一个地址。"""

    return tensor.untyped_storage().data_ptr()


@dataclass
class SavedTensorEvent:
    """一次 autograd 保存事件的可序列化描述。"""

    index: int
    shape: list[int]
    stride: list[int]
    dtype: str
    numel: int
    bytes: int
    storage_bytes: int
    storage_pointer: int
    storage_offset: int
    grad_fn: str | None
    is_parameter_storage: bool


class SavedTensorRecorder:
    """记录 autograd 保存/加载张量，并排除参数底层 storage。"""

    def __init__(self, parameters: Iterable[nn.Parameter]):
        self.parameter_storage_pointers = {storage_pointer(parameter) for parameter in parameters}
        self.saved: list[SavedTensorEvent] = []
        self.loaded_storage_pointers: list[int] = []

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """保存时采集元数据，原样返回 tensor 以保持 autograd 语义。"""

        pointer = storage_pointer(tensor)
        self.saved.append(
            SavedTensorEvent(
                index=len(self.saved),
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                dtype=str(tensor.dtype),
                numel=tensor.numel(),
                bytes=tensor.numel() * tensor.element_size(),
                storage_bytes=tensor.untyped_storage().nbytes(),
                storage_pointer=pointer,
                storage_offset=tensor.storage_offset(),
                grad_fn=type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None,
                is_parameter_storage=pointer in self.parameter_storage_pointers,
            )
        )
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        """加载时只记录 storage，原样返回。"""

        self.loaded_storage_pointers.append(storage_pointer(tensor))
        return tensor

    def summary(self) -> dict[str, Any]:
        """汇总事件字节、唯一 storage 字节和最常见的 shape/dtype。"""

        non_parameter_events = [event for event in self.saved if not event.is_parameter_storage]
        unique_storages: dict[int, int] = {}
        for event in non_parameter_events:
            unique_storages[event.storage_pointer] = max(
                unique_storages.get(event.storage_pointer, 0), event.storage_bytes
            )
        signatures = Counter((tuple(event.shape), event.dtype, event.grad_fn) for event in non_parameter_events)
        top_signatures = [
            {
                "shape": list(shape),
                "dtype": dtype,
                "grad_fn": grad_fn,
                "count": count,
            }
            for (shape, dtype, grad_fn), count in signatures.most_common(12)
        ]
        return {
            "saved_event_count": len(self.saved),
            "non_parameter_saved_event_count": len(non_parameter_events),
            "loaded_event_count": len(self.loaded_storage_pointers),
            "non_parameter_event_bytes": sum(event.bytes for event in non_parameter_events),
            "non_parameter_event_mib": sum(event.bytes for event in non_parameter_events) / MIB,
            "unique_non_parameter_storage_count": len(unique_storages),
            "unique_non_parameter_storage_bytes": sum(unique_storages.values()),
            "unique_non_parameter_storage_mib": sum(unique_storages.values()) / MIB,
            "top_signatures": top_signatures,
        }


def make_component(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, torch.Tensor]:
    """构造 RMSNorm 或单个 TransformerBlock 及其输入。"""

    if args.component == "rmsnorm":
        with torch.device(device):
            module: nn.Module = RMSNorm(args.d_model)
    else:
        with torch.device(device):
            positional_encoder = RotaryEmbedding(
                context_length=args.sequence_length,
                dim=args.d_model // args.num_heads,
            )
            module = TransformerBlock(
                d_model=args.d_model,
                d_ff=args.d_ff,
                num_heads=args.num_heads,
                positional_encoder=positional_encoder,
            )
    module = module.to(dtype=torch.float32)
    if args.compile:
        module = torch.compile(module, fullgraph=True)
    inputs = torch.randn(
        (args.batch_size, args.sequence_length, args.d_model),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    return module, inputs


def run_saved_tensors(args: argparse.Namespace) -> dict[str, Any]:
    """执行一次 forward/backward 并记录 autograd residual。"""

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # A2-K 要求在第一次 CUDA 分配之前设置 23 GiB allocator 上限。
    allocator = configure_cuda_allocator(device, args.allocator_limit_gib)
    module, inputs = make_component(args, device)
    for _ in range(args.warmup):
        warmup_output = module(inputs)
        warmup_output.float().sum().backward()
        inputs.grad = None
        for parameter in module.parameters():
            parameter.grad = None
        del warmup_output
    gc.collect()
    synchronize(device)
    recorder = SavedTensorRecorder(module.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.autograd.graph.saved_tensors_hooks(recorder.pack, recorder.unpack):
        output, forward_ms = elapsed_ms(device, lambda: module(inputs))
        memory_after_forward = cuda_memory_mib(device)
        _, backward_ms = elapsed_ms(device, lambda: output.float().sum().backward())

    return {
        "event": "saved_tensors_experiment",
        "status": "passed",
        "component": args.component,
        "compiled": args.compile,
        "warmup": args.warmup,
        "device": str(device),
        "allocator": allocator,
        "environment": hardware_metadata(device),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "input_activation_mib": inputs.numel() * inputs.element_size() / MIB,
        "timing_ms": {"forward": forward_ms, "backward": backward_ms},
        "memory_after_forward": memory_after_forward,
        "memory_after_backward": cuda_memory_mib(device),
        "summary": recorder.summary(),
        "events": [asdict(event) for event in recorder.saved] if args.include_events else None,
    }


class CheckpointedTransformerStack(nn.Module):
    """按固定 group size 对连续 TransformerBlock 做非嵌套 checkpoint。"""

    def __init__(
        self,
        *,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        context_length: int,
        checkpoint_strategy: str,
        checkpoint_group_size: int,
        compile_blocks: bool,
    ):
        super().__init__()
        self.checkpoint_strategy = checkpoint_strategy
        self.checkpoint_group_size = checkpoint_group_size
        positional_encoder = RotaryEmbedding(context_length=context_length, dim=d_model // num_heads)
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            block: nn.Module = TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                positional_encoder=positional_encoder,
            )
            if compile_blocks:
                block = torch.compile(block, fullgraph=True)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _run_group(hidden: torch.Tensor, layers: tuple[nn.Module, ...]) -> torch.Tensor:
        """顺序执行一个 checkpoint group 内的 block。"""

        for layer in layers:
            hidden = layer(hidden)
        return hidden

    def _run_layer_range(self, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """执行半开区间 ``[start, end)``，供递归 checkpoint 作为纯函数使用。"""

        for index in range(start, end):
            hidden = self.layers[index](hidden)
        return hidden

    def _run_nested(self, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """平衡二叉递归 checkpoint，对应题目 (a) 的低峰值策略。

        当区间不大于 leaf size 时直接执行。更大区间一分为二，并分别把左右
        子区间包装成 checkpoint。初次 forward 只保存少量边界；backward 重算
        某个子区间时，再展开下一层 checkpoint。递归深度约为 log2(N)，因此
        峰值边界激活为 O(log N)，代价是每层可能在多个递归层次被重算，计算量
        约为 O(N log N)。题目 (b) 禁止嵌套，所以正式硬件扫描仍使用 groups。
        """

        if end - start <= self.checkpoint_group_size:
            return self._run_layer_range(hidden, start, end)
        middle = start + (end - start) // 2
        hidden = checkpoint(self._run_nested, hidden, start, middle, use_reentrant=False)
        return checkpoint(self._run_nested, hidden, middle, end, use_reentrant=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """group size 为 0 时不 checkpoint，否则每组保存一个入口激活。"""

        if self.checkpoint_strategy == "none" or self.checkpoint_group_size == 0:
            return self._run_group(hidden, tuple(self.layers))

        if self.checkpoint_strategy == "nested":
            return self._run_nested(hidden, 0, len(self.layers))

        group_size = self.checkpoint_group_size
        for start in range(0, len(self.layers), group_size):
            # 显式按索引组装 tuple，避免不同 PyTorch 版本对 ModuleList 切片
            # 返回类型的差异进入 checkpoint 的非 Tensor 参数。
            layers = tuple(self.layers[index] for index in range(start, min(start + group_size, len(self.layers))))
            hidden = checkpoint(self._run_group, hidden, layers, use_reentrant=False)
        return hidden


def run_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """测量一个 checkpoint group size 的 forward/backward 峰值显存。"""

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.checkpoint_group_size < 0:
        raise ValueError("checkpoint group size must be >= 0")
    if args.checkpoint_group_size > args.num_layers:
        raise ValueError("checkpoint group size cannot exceed num_layers")
    if args.checkpoint_strategy == "nested" and args.checkpoint_group_size == 0:
        raise ValueError("nested checkpoint strategy requires a positive leaf group size")

    # 必须早于 TransformerStack 和输入的 CUDA allocation。
    allocator = configure_cuda_allocator(device, args.allocator_limit_gib)
    if args.precision not in ("fp32", "bf16-mixed"):
        raise ValueError("checkpoint precision must be fp32 or bf16-mixed")

    torch.manual_seed(args.seed)
    # xl stack 约有数十亿参数。直接在目标设备构造可避免先占一份巨大 CPU
    # 参数、再复制到 GPU 的瞬时双份内存。
    with torch.device(device):
        model = CheckpointedTransformerStack(
            num_layers=args.num_layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_heads=args.num_heads,
            context_length=args.sequence_length,
            checkpoint_strategy=args.checkpoint_strategy,
            checkpoint_group_size=args.checkpoint_group_size,
            compile_blocks=args.compile,
        )
    model = model.to(dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    after_model_init = cuda_memory_mib(device)
    inputs = torch.randn(
        (args.batch_size, args.sequence_length, args.d_model),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    def train_step() -> tuple[torch.Tensor, torch.Tensor, bool]:
        """执行一次完整训练步，并在 optimizer.step 前检查梯度有限性。"""

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            output = model(inputs)
            # loss 归约显式使用 FP32，避免 BF16 平方和平均造成额外误差。
            loss = output.float().square().mean()
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        inputs.grad = None
        return output, loss, gradients_finite

    # torch.compile 在首次 forward/backward 时才真正生成 CUDA 代码。先运行完整
    # warmup，再释放输出和梯度，正式峰值便只反映 checkpoint 策略本身。
    for _ in range(args.warmup):
        warmup_output, warmup_loss, _warmup_gradients_finite = train_step()
        # loss 仍然持有计算图的根引用；warm-up 最后一轮也必须显式释放，
        # 否则正式测量开始前可能还残留一个 output/loss 对象。
        del warmup_output, warmup_loss, _warmup_gradients_finite
    gc.collect()
    synchronize(device)

    before_forward = cuda_memory_mib(device)

    samples: list[float] = []
    peak_allocated_samples: list[float] = []
    peak_reserved_samples: list[float] = []
    output: torch.Tensor | None = None
    last_output_summary: dict[str, float] | None = None
    last_output_finite = True
    gradients_finite = True
    for _ in range(args.repeats):
        if device.type == "cuda":
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter_ns()
        output, _loss, current_gradients_finite = train_step()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
        gradients_finite = gradients_finite and current_gradients_finite
        after_sample = cuda_memory_mib(device)
        if after_sample is not None:
            peak_allocated_samples.append(after_sample["peak_allocated_mib"])
            peak_reserved_samples.append(after_sample["peak_reserved_mib"])
        last_output_finite = bool(torch.isfinite(output).all().item())
        last_output_summary = {
            "sum": float(output.detach().float().sum().cpu()),
            "mean": float(output.detach().float().mean().cpu()),
        }
        del output, _loss

    after_backward = cuda_memory_mib(device)
    after_forward = None

    baseline_allocated = before_forward["allocated_mib"] if before_forward is not None else None
    peak_allocated = max(peak_allocated_samples) if peak_allocated_samples else None
    group_count = (
        0
        if args.checkpoint_strategy == "none" or args.checkpoint_group_size == 0
        else math.ceil(args.num_layers / args.checkpoint_group_size)
    )
    return {
        "event": "checkpoint_experiment",
        "status": "passed",
        "device": str(device),
        "compiled_blocks": args.compile,
        "allocator": allocator,
        "environment": hardware_metadata(device),
        "precision": args.precision,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "checkpoint_strategy": args.checkpoint_strategy,
        "checkpoint_group_size": args.checkpoint_group_size,
        "checkpoint_group_count": group_count,
        "single_checkpoint_activation_mib": inputs.numel() * inputs.element_size() / MIB,
        "timing_ms": {
            "train_step": summarize_samples(samples),
            "forward_backward": summarize_samples(samples)["mean_ms"],
        },
        "memory": {
            "after_model_init": after_model_init,
            "before_forward": before_forward,
            "after_forward": after_forward,
            "peak_allocated_samples_mib": peak_allocated_samples,
            "peak_reserved_samples_mib": peak_reserved_samples,
            "after_backward": after_backward,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": max(peak_reserved_samples) if peak_reserved_samples else None,
            "peak_increment_over_forward_baseline_mib": (
                peak_allocated - baseline_allocated
                if peak_allocated is not None and baseline_allocated is not None
                else None
            ),
        },
        "output_finite": last_output_finite,
        "output_summary": last_output_summary,
        "gradient_finite": gradients_finite,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "optimizer": "AdamW",
    }


def oom_result(args: argparse.Namespace, error: torch.cuda.OutOfMemoryError) -> dict[str, Any]:
    """将 OOM 保存为有效实验观察。"""

    return {
        "event": f"{args.experiment}_experiment",
        "status": "oom",
        "device": args.device,
        "component": getattr(args, "component", None),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "num_layers": getattr(args, "num_layers", None),
        "checkpoint_group_size": getattr(args, "checkpoint_group_size", None),
        "checkpoint_strategy": getattr(args, "checkpoint_strategy", None),
        "compiled": args.compile,
        "precision": args.precision,
        "repeats": args.repeats,
        "allocator_limit_gib": args.allocator_limit_gib,
        "error": str(error),
    }


def add_shared_shape_args(parser: argparse.ArgumentParser) -> None:
    """为两个实验注册统一 shape 参数。"""

    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16-mixed"), default="fp32")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-oom", action="store_true")
    parser.add_argument(
        "--allocator-limit-gib",
        type=float,
        default=None,
        help="正式 4090 显存实验使用 23；必须早于模型创建",
    )


def parse_args() -> argparse.Namespace:
    """解析 saved-tensors 或 checkpoint 子命令。"""

    parser = argparse.ArgumentParser(description="Run CS336 single-GPU memory experiments.")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    saved_parser = subparsers.add_parser("saved-tensors")
    add_shared_shape_args(saved_parser)
    saved_parser.add_argument("--component", choices=("rmsnorm", "block"), default="rmsnorm")
    saved_parser.add_argument("--include-events", action="store_true")

    checkpoint_parser = subparsers.add_parser("checkpoint")
    add_shared_shape_args(checkpoint_parser)
    checkpoint_parser.add_argument("--num-layers", type=int, default=32)
    checkpoint_parser.add_argument(
        "--checkpoint-strategy",
        choices=("none", "groups", "nested"),
        default="groups",
    )
    checkpoint_parser.add_argument("--checkpoint-group-size", type=int, default=1)
    checkpoint_parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_result(result: dict[str, Any], output: Path | None) -> None:
    """打印结果并可选追加到 JSONL。"""

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> int:
    """执行指定实验；OOM 是否返回成功由 ``--allow-oom`` 决定。"""

    args = parse_args()
    try:
        result = run_saved_tensors(args) if args.experiment == "saved-tensors" else run_checkpoint(args)
        exit_code = 0
    except torch.cuda.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = oom_result(args, error)
        exit_code = 0 if args.allow_oom else 2
    write_result(result, args.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
