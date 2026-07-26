"""A2 Profiling and Benchmarking 的统一训练步测量工具。

本文件覆盖讲义第一部分会反复使用的测量能力：

* 表 1 的 ``small``、``medium``、``large``、``xl`` 和 ``10b`` 配置；
* forward、backward、optimizer、forward+backward 和完整训练步；
* FP32 与 BF16 autocast 混合精度；
* eager 与 ``torch.compile`` 模型；
* warmup、逐次原始样本、统计量、CUDA 峰值显存和可选 memory snapshot；
* 将 CUDA OOM 保存成结构化结果，而不是让参数扫描悄悄漏掉失败配置。

这里不实现 FlashAttention、DDP、优化器分片或 FSDP。它只提供可信的实验
边界，供后续实现使用同一套口径比较正确性、速度和显存。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax as basics_softmax
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k.configs import MODEL_CONFIGS, ModelConfig, resolve_model_config
from cs336_systems.a2k.experiment_utils import (
    configure_cuda_allocator,
    hardware_metadata,
    summarize_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


@contextmanager
def nvtx_range(name: str, device: torch.device) -> Iterator[None]:
    """在 Nsight 时间线上标记训练语义阶段，CPU 运行时退化为空操作。"""

    enabled = device.type == "cuda" and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def synchronize(device: torch.device) -> None:
    """在计时边界等待 GPU 完成，避免只测到异步 kernel 的提交时间。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now_ms() -> float:
    """返回高分辨率墙钟时间，单位为毫秒。"""

    return time.perf_counter_ns() / 1_000_000


def summarize(values: list[float]) -> dict[str, float]:
    """保留均值、样本标准差、中位数和范围，便于识别冷启动与异常值。"""

    if not values:
        raise ValueError("at least one latency sample is required")
    # 保留旧版函数名，避免已有外部脚本失效；正式结果使用共享工具补充
    # CV 和 p20/p50/p80。
    summary = summarize_samples(values)
    summary["median_ms"] = summary["p50_ms"]
    return summary


def resolve_precision(name: str) -> str:
    """把旧参数名称归一化为明确的 FP32 或 BF16 mixed precision。"""

    aliases = {
        "float32": "fp32",
        "fp32": "fp32",
        "bfloat16": "bf16-mixed",
        "bf16": "bf16-mixed",
        "bf16-mixed": "bf16-mixed",
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported precision: {name}; choose fp32 or bf16-mixed") from exc


def autocast_context(device: torch.device, precision: str):
    """返回本轮 forward 使用的精度上下文。

    混合精度不把长期保存的模型参数直接改成 BF16。参数和 AdamW 状态仍为
    FP32；autocast 只让适合的 CUDA 算子在 forward/backward 中采用 BF16，
    更接近真实训练系统的 mixed precision，而不是“全模型强制降精度”。
    """

    if precision == "bf16-mixed":
        if device.type != "cuda":
            raise ValueError("BF16 mixed precision is only enabled for CUDA experiments")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def memory_stats(device: torch.device) -> dict[str, float] | None:
    """读取当前与峰值 CUDA allocator 指标，统一转换为 MiB。"""

    if device.type != "cuda":
        return None
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / MIB,
        "reserved_mib": torch.cuda.memory_reserved(device) / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
    }


def parameter_bytes(model: torch.nn.Module) -> int:
    """统计去重后的参数存储字节数，避免 tied weights 被重复计算。"""

    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        storage_id = parameter.untyped_storage().data_ptr()
        if storage_id not in seen:
            seen.add(storage_id)
            total += parameter.numel() * parameter.element_size()
    return total


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """带细粒度 NVTX 标记、但数学结果不变的标准 Attention。

    Nsight 只能看到底层 CUDA kernel 名称。只看诸如 ``ampere_sgemm`` 的名字，
    很难判断它属于 QK^T、softmax 还是最后的 PV。正式 profile 可用
    ``--annotate-attention`` 临时替换 A1 实现，在时间线上增加三个语义区域。
    这里刻意保留标准的二次复杂度实现，不能把它误当成 FlashAttention。
    """

    device = Q.device
    with nvtx_range("attention/scores", device):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
    if mask is not None:
        with nvtx_range("attention/mask", device):
            scores = torch.where(mask, scores, float("-inf"))
    with nvtx_range("attention/softmax", device):
        # 使用 A1 原始 softmax 的 max/exp/sum/div 算子分解。若改成框架融合的
        # torch.softmax，NVTX 标注本身就会改变被分析的 kernel，结论不再可信。
        probabilities = basics_softmax(scores, dim=-1)
    with nvtx_range("attention/value", device):
        return torch.matmul(probabilities, V)


def build_model(
    config: ModelConfig,
    device: torch.device,
    *,
    compile_model: bool,
    compile_mode: str,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """构建 FP32 master model、AdamW，并可选编译模型 forward。"""

    # 10B 配置若先在 CPU 构造再搬到 GPU，会短暂占用约 40 GiB 主存，还会让
    # 一个注定 OOM 的 CUDA case 在数据传输上浪费很久。默认设备上下文使参数
    # 直接出生在目标设备；小模型的 CPU 冒烟测试仍走完全相同的代码路径。
    with torch.device(device):
        base_model = BasicsTransformerLM(**config.to_dict())
    base_model = base_model.to(dtype=torch.float32)
    model_parameter_bytes = parameter_bytes(base_model)
    optimizer = AdamW(base_model.parameters(), lr=1e-3)

    # optimizer 持有原始 Parameter；torch.compile 返回的模块仍转发到同一组参数。
    # 先创建 optimizer 可以让“是否 compile”只改变执行图，不改变更新的张量。
    model: torch.nn.Module = base_model
    if compile_model:
        model = torch.compile(base_model, mode=compile_mode)
    return model, optimizer, model_parameter_bytes


def register_block_nvtx_hooks(model: torch.nn.Module, device: torch.device) -> list[torch.utils.hooks.RemovableHandle]:
    """给每个 TransformerBlock forward 添加可嵌套的 Nsight 标签。

    这些 hook 只用于 eager profiling。反向传播的算子由 Nsight 的 PyTorch
    autograd NVTX 支持标记；forward hook 则让我们能直接筛选单层分配与 kernel。
    """

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for index, layer in enumerate(getattr(model, "layers", ())):
        label = f"transformer_block_{index}"

        def push_range(_module, _inputs, *, range_label=label):
            if device.type == "cuda":
                torch.cuda.nvtx.range_push(range_label)

        def pop_range(_module, _inputs, _output):
            if device.type == "cuda":
                torch.cuda.nvtx.range_pop()

        handles.append(layer.register_forward_pre_hook(push_range))
        handles.append(layer.register_forward_hook(pop_range))
    return handles


def make_batch(config: ModelConfig, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """在计时外生成固定形状的随机 token 与目标。"""

    inputs = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
    return inputs, targets


def forward_only(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    """执行模型 forward；精度上下文只覆盖模型计算。"""

    with nvtx_range("forward", device), autocast_context(device, precision):
        return model(inputs)


def forward_and_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 forward 和稳定的 FP32 loss 归约。"""

    logits = forward_only(model, inputs, device, precision)
    with nvtx_range("loss", device):
        # A1 的 cross_entropy 由基础 PyTorch 运算组成，不在 autocast 的稳定算子
        # 白名单中。显式升为 FP32，避免 BF16 exp/sum 归约放大数值误差。
        loss = cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1))
    return logits, loss


def elapsed_ms(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    """在统一同步边界下执行函数并测量墙钟时间。"""

    synchronize(device)
    start = now_ms()
    result = function()
    synchronize(device)
    return result, now_ms() - start


def run_iteration(
    args: argparse.Namespace,
    config: ModelConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    """执行一个测量样本，并明确该模式计时区间内包含哪些工作。"""

    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    before = memory_stats(device)

    logits: torch.Tensor
    loss: torch.Tensor | None = None

    if args.mode == "inference":
        with torch.no_grad(), nvtx_range("forward_only", device):
            logits, latency_ms = elapsed_ms(device, lambda: forward_only(model, inputs, device, precision))

    elif args.mode == "forward":
        # A2-P 的 forward 只回答“模型前向本身多快”，不应把 autograd
        # 保存图的开销混进来；需要保留计算图的组合口径由 forward_backward
        # 和 train_step 单独测量。
        with torch.no_grad():
            logits, latency_ms = elapsed_ms(device, lambda: forward_only(model, inputs, device, precision))

    elif args.mode == "backward":
        optimizer.zero_grad(set_to_none=True)
        logits, loss = forward_and_loss(model, inputs, targets, config.vocab_size, device, precision)
        with nvtx_range("backward", device):
            _, latency_ms = elapsed_ms(device, loss.backward)
        optimizer.zero_grad(set_to_none=True)

    elif args.mode == "optimizer":
        # optimizer step 依赖已经存在的梯度，因此 forward/backward 在计时外准备。
        optimizer.zero_grad(set_to_none=True)
        logits, loss = forward_and_loss(model, inputs, targets, config.vocab_size, device, precision)
        loss.backward()
        synchronize(device)
        with nvtx_range("optimizer", device):
            _, latency_ms = elapsed_ms(device, optimizer.step)
        optimizer.zero_grad(set_to_none=True)

    elif args.mode == "forward_backward":
        def forward_backward() -> tuple[torch.Tensor, torch.Tensor]:
            optimizer.zero_grad(set_to_none=True)
            current_logits, current_loss = forward_and_loss(
                model, inputs, targets, config.vocab_size, device, precision
            )
            with nvtx_range("backward", device):
                current_loss.backward()
            return current_logits, current_loss

        (logits, loss), latency_ms = elapsed_ms(device, forward_backward)
        optimizer.zero_grad(set_to_none=True)

    else:
        def full_step() -> tuple[torch.Tensor, torch.Tensor]:
            step_label = "train_step" if args.mode == "train_step" else "full_step"
            with nvtx_range(step_label, device):
                with nvtx_range("zero_grad_begin", device):
                    optimizer.zero_grad(set_to_none=True)
                current_logits, current_loss = forward_and_loss(
                    model, inputs, targets, config.vocab_size, device, precision
                )
                with nvtx_range("backward", device):
                    current_loss.backward()
                with nvtx_range("optimizer", device):
                    optimizer.step()
                with nvtx_range("zero_grad_end", device):
                    optimizer.zero_grad(set_to_none=True)
            return current_logits, current_loss

        (logits, loss), latency_ms = elapsed_ms(device, full_step)

    after = memory_stats(device)
    return {
        "latency_ms": latency_ms,
        "loss": float(loss.detach().cpu()) if loss is not None else None,
        "logits_dtype": str(logits.dtype),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "memory_before": before,
        "memory_after": after,
    }


@contextmanager
def optional_memory_history(snapshot_path: Path | None, device: torch.device) -> Iterator[None]:
    """只在请求时记录一次 PyTorch allocator 历史并导出 snapshot。"""

    if snapshot_path is None:
        yield
        return
    if device.type != "cuda":
        raise ValueError("memory snapshots require a CUDA device")

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    try:
        yield
        synchronize(device)
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def resolve_config_from_args(args: argparse.Namespace) -> ModelConfig:
    """把命名模型与所有可选字段覆盖解析为最终配置。"""

    return resolve_model_config(
        args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """执行 warmup、正式样本、显存采集与统计汇总。"""

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    precision = resolve_precision(args.dtype)
    config = resolve_config_from_args(args)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    # 只有显式要求正式 4090 口径时才设置上限，避免改变旧版 benchmark
    # 的历史结果。这个调用发生在模型和输入创建之前，满足 allocator guard
    # 必须早于第一次 CUDA 大分配的要求。
    allocator = configure_cuda_allocator(device, args.allocator_limit_gib)

    original_attention = basics_model.scaled_dot_product_attention
    if args.annotate_attention:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    try:
        model, optimizer, model_parameter_bytes = build_model(
            config,
            device,
            compile_model=args.compile_model,
            compile_mode=args.compile_mode,
        )
    finally:
        # 模型 forward 在运行时从 ``cs336_basics.model`` 的全局名称查找函数，
        # 所以构造完成后仍需保持替换。这里只在构造失败时立即恢复；正常路径在
        # 全部测量完成后恢复，避免影响同一解释器中的后续调用。
        if "model" not in locals():
            basics_model.scaled_dot_product_attention = original_attention
    block_hook_handles = register_block_nvtx_hooks(model, device) if args.annotate_blocks else []
    inputs, targets = make_batch(config, args.batch_size, device)

    # torch.compile 的第一次调用可能包含图捕获、代码生成和缓存建立。
    # 对模型级 compile 对照，先把这一次完整记录为 cold start，再让正式
    # warm-up/measurement 只反映已经生成代码后的 steady-state。
    compile_cold_start_ms = None
    if args.compile_model:
        cold_sample = run_iteration(args, config, model, optimizer, inputs, targets, device, precision)
        compile_cold_start_ms = cold_sample["latency_ms"]

    if args.warmup:
        with nvtx_range("profile/warmup", device):
            for _ in range(args.warmup):
                run_iteration(args, config, model, optimizer, inputs, targets, device, precision)

    samples: list[dict[str, Any]] = []
    # 该外层标签只覆盖正式样本，不覆盖 warmup。Nsight 使用 NVTX capture-range
    # 后可从这里开始采集，避免编译和冷启动把正式 kernel 统计淹没。
    with nvtx_range("profile/measure", device):
        for repeat_index in range(args.repeats):
            # snapshot 只包住第一个正式样本；其余样本用于统计，不重复生成大文件。
            snapshot_path = args.memory_snapshot if repeat_index == 0 else None
            with optional_memory_history(snapshot_path, device):
                samples.append(run_iteration(args, config, model, optimizer, inputs, targets, device, precision))

    basics_model.scaled_dot_product_attention = original_attention
    for handle in block_hook_handles:
        handle.remove()

    timing = summarize([sample["latency_ms"] for sample in samples])
    peak_allocated = [
        sample["memory_after"]["peak_allocated_mib"]
        for sample in samples
        if sample["memory_after"] is not None
    ]
    peak_reserved = [
        sample["memory_after"]["peak_reserved_mib"]
        for sample in samples
        if sample["memory_after"] is not None
    ]
    return {
        "event": "benchmark_training_step",
        "status": "passed",
        "seed": args.seed,
        "device": str(device),
        "precision": precision,
        "mode": "train_step" if args.mode == "train_step" else args.mode,
        "model_size": args.model_size,
        "model": config.to_dict(),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "compile_model": args.compile_model,
        "compile_mode": args.compile_mode if args.compile_model else None,
        "compile_cold_start_ms": compile_cold_start_ms,
        "annotate_attention": args.annotate_attention,
        "annotate_blocks": args.annotate_blocks,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_storage_mib": model_parameter_bytes / MIB,
        "allocator": allocator,
        "environment": {
            **hardware_metadata(device),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "samples": samples,
        "timing": timing,
        "peak_memory": {
            "max_allocated_mib": max(peak_allocated) if peak_allocated else None,
            "max_reserved_mib": max(peak_reserved) if peak_reserved else None,
        },
        "timing_boundary": {
            "inference": "model(inputs) under torch.no_grad()",
            "forward": "torch.no_grad() + model(inputs), excluding loss/backward/optimizer",
            "backward": "loss.backward() after forward/loss preparation outside the timer",
            "optimizer": "optimizer.step() after gradients are prepared outside the timer",
            "forward_backward": "zero_grad + forward + FP32 loss + backward",
            "full_step": "zero_grad + forward + FP32 loss + backward + optimizer.step + zero_grad",
            "train_step": "zero_grad + forward + FP32 loss + backward + optimizer.step + zero_grad",
        }[args.mode],
        "all_logits_finite": all(sample["logits_finite"] for sample in samples),
        "memory_snapshot": str(args.memory_snapshot) if args.memory_snapshot else None,
    }


def oom_result(args: argparse.Namespace, error: torch.cuda.OutOfMemoryError) -> dict[str, Any]:
    """把 OOM 变成矩阵扫描可以继续处理的一条结构化记录。"""

    config = resolve_config_from_args(args)
    return {
        "event": "benchmark_training_step",
        "status": "oom",
        "device": args.device,
        "precision": resolve_precision(args.dtype),
        "mode": args.mode,
        "model_size": args.model_size,
        "model": config.to_dict(),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "compile_model": args.compile_model,
        "compile_cold_start_ms": None,
        "error": str(error),
    }


def parse_args() -> argparse.Namespace:
    """解析单个 benchmark 配置；正式参数矩阵由独立 runner 反复调用。"""

    parser = argparse.ArgumentParser(description="Benchmark one CS336 Transformer configuration.")
    parser.add_argument(
        "--mode",
        choices=("inference", "forward", "backward", "optimizer", "forward_backward", "full_step", "train_step"),
        default="full_step",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", default="float32", help="float32/fp32 or bfloat16/bf16-mixed")
    parser.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), default="tiny")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--allocator-limit-gib",
        type=float,
        default=None,
        help="在首次 CUDA 分配前设置 PyTorch allocator 上限，例如正式 4090 使用 23",
    )
    parser.add_argument(
        "--annotate-attention",
        action="store_true",
        help="replace standard attention with an equivalent NVTX-annotated implementation",
    )
    parser.add_argument(
        "--annotate-blocks",
        action="store_true",
        help="add one NVTX range around each eager TransformerBlock forward",
    )
    parser.add_argument("--memory-snapshot", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-oom", action="store_true")
    return parser.parse_args()


def write_result(result: dict[str, Any], output: Path | None) -> None:
    """输出人类可读 JSON，并可选追加到 JSONL 主结果。"""

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> int:
    """执行一个配置；OOM 是否视为进程失败由 ``--allow-oom`` 控制。"""

    args = parse_args()
    try:
        result = run_benchmark(args)
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
