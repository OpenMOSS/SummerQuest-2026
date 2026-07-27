import copy
import json
import math
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层全连接输出
        fc1_output = self.fc1(x)

        # ReLU 输出
        relu_output = self.relu(fc1_output)

        # LayerNorm 输出
        ln_output = self.ln(relu_output)

        # 最终输出，也叫 logits
        logits = self.fc2(ln_output)

        return logits, {
            "fc1_output": fc1_output,
            "ln_output": ln_output,
            "logits": logits,
        }


def dtype_name(value):
    """
    将 torch.dtype 转换成字符串，例如：
    torch.float32 -> float32
    torch.bfloat16 -> bfloat16
    """
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    return str(value)


def run_one_mode(
    mode_name,
    base_model,
    x,
    target,
    warmup_steps=5,
    measure_steps=20,
):
    """
    运行一种模式：

    mode_name:
        fp32 或 bf16

    base_model:
        相同初始化的模型，用于保证两个实验起点一致
    """

    device = x.device

    # 使用同一个初始模型参数
    model = copy.deepcopy(base_model).to(device)
    model.train()

    # BF16 autocast 不需要 GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    if mode_name == "fp32":
        autocast_context = lambda: nullcontext()
    elif mode_name == "bf16":
        autocast_context = lambda: torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unknown mode: {mode_name}")

    def train_step():
        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            logits, intermediate = model(x)

            # 使用 MSE loss。
            # target 是 FP32，PyTorch 会根据算子规则决定 loss 的计算类型。
            loss = F.mse_loss(logits, target)

        loss.backward()
        optimizer.step()

        return loss, intermediate

    # CUDA 是异步执行的，计时前先同步
    torch.cuda.synchronize()

    # 预热阶段，不纳入正式统计
    for _ in range(warmup_steps):
        train_step()

    torch.cuda.synchronize()

    step_times_ms = []
    peak_allocated_mb = []
    peak_reserved_mb = []

    recorded_dtypes = None
    final_loss = None

    for step in range(measure_steps):
        # 清除上一次迭代的显存峰值记录
        torch.cuda.reset_peak_memory_stats(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        loss, intermediate = train_step()

        end_event.record()

        # 必须同步，否则读取到的时间可能不完整
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)

        step_times_ms.append(float(elapsed_ms))
        peak_allocated_mb.append(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
        )
        peak_reserved_mb.append(
            float(torch.cuda.max_memory_reserved(device) / 1024**2)
        )

        final_loss = float(loss.detach().float().cpu())

        # 只记录第一次正式测量迭代的数据类型
        if recorded_dtypes is None:
            first_parameter = next(model.parameters())
            first_gradient = model.fc1.weight.grad

            recorded_dtypes = {
                "model_parameter": dtype_name(first_parameter.dtype),
                "first_feed_forward_output": dtype_name(
                    intermediate["fc1_output"].dtype
                ),
                "layer_norm_output": dtype_name(
                    intermediate["ln_output"].dtype
                ),
                "logits": dtype_name(intermediate["logits"].dtype),
                "loss": dtype_name(loss.dtype),
                "gradient": dtype_name(first_gradient.dtype),
            }

    mean_time_ms = statistics.mean(step_times_ms)
    std_time_ms = statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0

    return {
        "mode": mode_name,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "mean_step_time_ms": mean_time_ms,
        "std_step_time_ms": std_time_ms,
        "min_step_time_ms": min(step_times_ms),
        "max_step_time_ms": max(step_times_ms),
        "coefficient_of_variation": (
            std_time_ms / mean_time_ms if mean_time_ms != 0 else math.nan
        ),
        "peak_memory_allocated_mb_mean": statistics.mean(peak_allocated_mb),
        "peak_memory_allocated_mb_max": max(peak_allocated_mb),
        "peak_memory_reserved_mb_mean": statistics.mean(peak_reserved_mb),
        "peak_memory_reserved_mb_max": max(peak_reserved_mb),
        "final_loss": final_loss,
        "dtypes": recorded_dtypes,
        "step_times_ms": step_times_ms,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")

    print("PyTorch version:", torch.__version__)
    print("CUDA device:", torch.cuda.get_device_name(device))
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Current GPU does not support BF16")

    # 固定随机种子，保证实验可复现
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 实验规模
    batch_size = 4
    context_length = 512
    in_features = 1024
    out_features = 1024

    # 将 batch 和 sequence 合并成样本维度
    num_samples = batch_size * context_length

    # 输入保持 FP32。
    # BF16 autocast 会在支持的 CUDA 算子内部自动选择 BF16。
    x = torch.randn(
        num_samples,
        in_features,
        device=device,
        dtype=torch.float32,
    )

    target = torch.randn(
        num_samples,
        out_features,
        device=device,
        dtype=torch.float32,
    )

    # 先在 CPU 上构造一个 FP32 初始模型，
    # 两种模式都从这个模型复制，保证起点相同。
    base_model = ToyModel(
        in_features=in_features,
        out_features=out_features,
    ).float()

    results = {
        "experiment": "ToyModel FP32 versus CUDA BF16 autocast",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "context_length": context_length,
        "in_features": in_features,
        "out_features": out_features,
        "input_dtype": "float32",
        "target_dtype": "float32",
        "warmup_steps": 5,
        "measure_steps": 20,
        "modes": [],
    }

    for mode_name in ["fp32", "bf16"]:
        print(f"\n===== Running {mode_name} =====")

        mode_result = run_one_mode(
            mode_name=mode_name,
            base_model=base_model,
            x=x,
            target=target,
            warmup_steps=5,
            measure_steps=20,
        )

        results["modes"].append(mode_result)

        print("mean step time:",
              f"{mode_result['mean_step_time_ms']:.3f} ms")
        print("std step time:",
              f"{mode_result['std_step_time_ms']:.3f} ms")
        print("peak allocated:",
              f"{mode_result['peak_memory_allocated_mb_max']:.2f} MiB")
        print("peak reserved:",
              f"{mode_result['peak_memory_reserved_mb_max']:.2f} MiB")
        print("final loss:",
              f"{mode_result['final_loss']:.6f}")
        print("dtypes:")
        for key, value in mode_result["dtypes"].items():
            print(f"  {key}: {value}")

    output_path = Path("results/mixed_precision.json")
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()