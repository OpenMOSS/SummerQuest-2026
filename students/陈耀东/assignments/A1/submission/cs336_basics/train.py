"""CS336 Transformer LM 训练入口。

该脚本把已经通过单元测试的模型、损失函数、优化器、学习率调度、
梯度裁剪、memmap 数据读取和 checkpoint 组合成完整训练循环。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data_utils import get_batch
from cs336_basics.nn_modules import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping_with_norm
from cs336_basics.optim import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint


def parse_bool(value: str) -> bool:
    """解析配置文件传入的显式布尔字符串。"""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值：{value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CS336 Transformer LM")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--normalization", choices=("pre", "post", "none"), default="pre")
    parser.add_argument("--positional-encoding", choices=("rope", "none"), default="rope")
    parser.add_argument("--ffn-type", choices=("swiglu", "silu"), default="swiglu")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead"),
        default="none",
    )
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument("--fail-on-non-finite", type=parse_bool, default=True)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    """解析设备参数；auto 优先使用 CUDA。"""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def synchronize_device(device: torch.device) -> None:
    """在 CUDA 计时时等待异步 kernel 完成，CPU 上为空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, amp: str):
    """创建训练和验证共用的 autocast 上下文。"""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp == "bf16",
    )


def open_token_data(path: str | os.PathLike) -> np.memmap:
    """以内存映射方式读取 uint16 token 文件。"""
    return np.memmap(path, dtype=np.uint16, mode="r")


def compute_loss(model: TransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算一个 batch 的 next-token 平均交叉熵。"""
    logits = model(x)
    return cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        y.reshape(-1),
    )


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
    amp: str,
) -> float:
    """在随机验证 batch 上估计平均 loss。"""
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        x, y = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=str(device),
        )
        with autocast_context(device, amp):
            losses.append(compute_loss(model, x, y).item())
    model.train()
    return float(np.mean(losses))


def write_log(log_path: Path, record: dict[str, object]) -> None:
    """追加一条 JSONL 训练日志。"""
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_log_progress(log_path: Path) -> tuple[int, float]:
    """读取最后一个有效日志点，用于恢复累计 step 和墙钟时间。"""
    if not log_path.is_file():
        return 0, 0.0

    last_record: dict[str, object] | None = None
    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            if line.strip():
                last_record = json.loads(line)

    if last_record is None:
        return 0, 0.0
    return int(last_record["step"]), float(last_record["wall_clock_sec"])


def write_json(path: Path, payload: dict[str, object]) -> None:
    """以稳定的 UTF-8 JSON 格式写出配置或实验汇总。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def cuda_peak_memory_bytes(device: torch.device) -> int | None:
    """返回当前 CUDA 设备的历史峰值已分配显存。"""
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def build_summary_base(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, object]:
    """构造完成和失败 run 共用的公开汇总字段。"""
    return {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "normalization": args.normalization,
        "positional_encoding": args.positional_encoding,
        "ffn_type": args.ffn_type,
        "batch_size": args.batch_size,
        "total_steps": args.max_iters,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "matmul_precision": args.matmul_precision,
        "compile_mode": args.compile_mode,
        "amp": args.amp,
    }


def main() -> None:
    args = parse_args()
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")
    if args.max_iters <= 0:
        raise ValueError("max_iters 必须大于 0")
    if min(args.log_interval, args.eval_interval, args.eval_iters, args.checkpoint_interval) <= 0:
        raise ValueError("日志、验证和 checkpoint 的 interval/iters 必须大于 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("bf16 AMP 当前只用于 CUDA benchmark")
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    last_logged_step, elapsed_offset = read_log_progress(log_path)
    if args.resume is None and last_logged_step > 0:
        raise FileExistsError("输出目录已有训练日志；新实验请使用新的 output_dir")

    config_path = args.output_dir / "config.json"
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config["resolved_device"] = str(device)
    write_json(
        config_path,
        config,
    )

    train_data = open_token_data(args.train_data)
    valid_data = open_token_data(args.valid_data)
    minimum_tokens = args.context_length + 1
    if len(train_data) < minimum_tokens or len(valid_data) < minimum_tokens:
        raise ValueError("训练集和验证集都必须至少包含 context_length + 1 个 token")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        normalization=args.normalization,
        positional_encoding=args.positional_encoding,
        ffn_type=args.ffn_type,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iteration = 0
    if args.resume is not None:
        start_iteration = load_checkpoint(
            src=args.resume,
            model=model,
            optimizer=optimizer,
        )
        if last_logged_step > start_iteration:
            raise ValueError("训练日志的最后 step 晚于 checkpoint，不能安全追加")
    if start_iteration >= args.max_iters:
        raise ValueError("checkpoint iteration 必须小于 max_iters")

    training_model: torch.nn.Module = model
    if args.compile_mode != "none":
        compile_kwargs = {} if args.compile_mode == "default" else {"mode": args.compile_mode}
        training_model = torch.compile(model, **compile_kwargs)

    run_start = time.perf_counter()
    last_validation_loss: float | None = None
    step_times: list[float] = []
    training_model.train()

    def fail_run(
        failure: str,
        completed_iteration: int,
        learning_rate: float,
        train_loss: float | None,
        grad_norm: float | None,
    ) -> None:
        """写入结构化失败证据后终止当前 run。"""
        wall_clock_sec = elapsed_offset + time.perf_counter() - run_start
        processed_tokens = completed_iteration * args.batch_size * args.context_length
        write_log(
            log_path,
            {
                "step": completed_iteration,
                "wall_clock_sec": wall_clock_sec,
                "train_loss": train_loss,
                "lr": learning_rate,
                "processed_tokens": processed_tokens,
                "grad_norm": grad_norm,
                "failure": failure,
            },
        )
        summary = build_summary_base(args, model, device)
        summary.update(
            {
                "status": "diverged",
                "failure": failure,
                "final_val_loss": last_validation_loss,
                "total_wall_clock_sec": wall_clock_sec,
                "processed_tokens": processed_tokens,
                "cuda_peak_memory_bytes": cuda_peak_memory_bytes(device),
            }
        )
        write_json(args.output_dir / "summary.json", summary)
        raise FloatingPointError(f"训练终止：{failure}")

    for iteration in range(start_iteration, args.max_iters):
        learning_rate = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        synchronize_device(device)
        step_start = time.perf_counter()
        x, y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=str(device),
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            loss = compute_loss(training_model, x, y)
        train_loss_value = float(loss.detach().item())
        if args.fail_on_non_finite and not math.isfinite(train_loss_value):
            fail_run(
                failure="non_finite_loss",
                completed_iteration=iteration + 1,
                learning_rate=learning_rate,
                train_loss=None,
                grad_norm=None,
            )
        loss.backward()
        grad_norm_tensor = gradient_clipping_with_norm(model.parameters(), args.grad_clip)
        grad_norm_value = float(grad_norm_tensor.detach().item())
        if args.fail_on_non_finite and not math.isfinite(grad_norm_value):
            fail_run(
                failure="non_finite_gradient_norm",
                completed_iteration=iteration + 1,
                learning_rate=learning_rate,
                train_loss=train_loss_value,
                grad_norm=None,
            )
        optimizer.step()
        synchronize_device(device)
        step_time_sec = time.perf_counter() - step_start
        step_times.append(step_time_sec)
        tokens_per_second = args.batch_size * args.context_length / max(step_time_sec, 1e-12)

        completed_iteration = iteration + 1
        should_evaluate = (
            completed_iteration == 1
            or completed_iteration % args.eval_interval == 0
            or completed_iteration == args.max_iters
        )
        validation_loss: float | None = None
        if should_evaluate:
            validation_loss = evaluate(
                model=training_model,
                dataset=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_iters=args.eval_iters,
                device=device,
                amp=args.amp,
            )
            last_validation_loss = validation_loss

        should_log = (
            completed_iteration == 1
            or completed_iteration % args.log_interval == 0
            or should_evaluate
            or completed_iteration == args.max_iters
        )
        if should_log:
            wall_clock_sec = elapsed_offset + time.perf_counter() - run_start
            record = {
                "step": completed_iteration,
                "wall_clock_sec": wall_clock_sec,
                "train_loss": train_loss_value,
                "lr": learning_rate,
                "processed_tokens": completed_iteration * args.batch_size * args.context_length,
                "step_time_sec": step_time_sec,
                "tokens_per_second": tokens_per_second,
                "grad_norm": grad_norm_value,
            }
            peak_memory = cuda_peak_memory_bytes(device)
            if peak_memory is not None:
                record["cuda_max_memory_bytes"] = peak_memory
            if validation_loss is not None:
                record["val_loss"] = validation_loss
            write_log(log_path, record)
            val_text = "" if validation_loss is None else f" val_loss={validation_loss:.4f}"
            print(
                f"step={completed_iteration} "
                f"train_loss={train_loss_value:.4f} "
                f"lr={learning_rate:.3e} "
                f"tokens_per_second={tokens_per_second:.1f} "
                f"grad_norm={grad_norm_value:.4f} "
                f"time={wall_clock_sec:.1f}s"
                f"{val_text}"
            )

        should_checkpoint = completed_iteration % args.checkpoint_interval == 0 or completed_iteration == args.max_iters
        if should_checkpoint:
            checkpoint_path = args.output_dir / f"checkpoint_{completed_iteration:07d}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_iteration,
                out=checkpoint_path,
            )

    total_wall_clock_sec = elapsed_offset + time.perf_counter() - run_start
    warmup_steps_to_exclude = 5 if len(step_times) >= 20 else 0
    steady_step_times = step_times[warmup_steps_to_exclude:]
    steady_step_mean = float(np.mean(steady_step_times))
    average_tokens_per_second = args.batch_size * args.context_length / steady_step_mean
    summary = build_summary_base(args, model, device)
    summary.update(
        {
            "status": "completed",
            "final_val_loss": last_validation_loss,
            "total_wall_clock_sec": total_wall_clock_sec,
            "processed_tokens": args.max_iters * args.batch_size * args.context_length,
            "training_step_wall_clock_sec": float(sum(step_times)),
            "average_step_time_sec": steady_step_mean,
            "average_tokens_per_second": average_tokens_per_second,
            "benchmark_warmup_steps_excluded": warmup_steps_to_exclude,
            "cuda_peak_memory_bytes": cuda_peak_memory_bytes(device),
        }
    )
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
