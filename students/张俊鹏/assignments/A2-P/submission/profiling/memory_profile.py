import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function


class TinyTransformerLM(nn.Module):
    """
    一个用于显存分析的小型 Transformer 语言模型。
    """

    def __init__(
        self,
        vocab_size=8192,
        context_length=512,
        d_model=512,
        nhead=8,
        num_layers=4,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        batch_size, sequence_length = tokens.shape

        positions = torch.arange(
            sequence_length,
            device=tokens.device,
        )

        with record_function("embedding"):
            x = self.token_embedding(tokens)
            x = x + self.position_embedding(positions)[None, :, :]

        with record_function("transformer"):
            x = self.transformer(x)

        with record_function("final_norm"):
            x = self.final_norm(x)

        with record_function("lm_head"):
            logits = self.lm_head(x)

        return logits


def memory_snapshot():
    """
    读取当前 CUDA 显存状态。
    """

    torch.cuda.synchronize()

    return {
        "allocated_mib": float(
            torch.cuda.memory_allocated() / 1024**2
        ),
        "reserved_mib": float(
            torch.cuda.memory_reserved() / 1024**2
        ),
        "max_allocated_mib": float(
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "max_reserved_mib": float(
            torch.cuda.max_memory_reserved() / 1024**2
        ),
    }


def begin_memory_stage():
    """
    开始一个新的显存统计阶段。

    reset_peak_memory_stats 只重置峰值记录，
    不会释放当前已经存在的显存。
    """

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def run_profiled_train_step(model, optimizer, tokens, targets):
    """
    执行一次带阶段标记的训练步骤。
    """

    optimizer.zero_grad(set_to_none=True)

    stage_stats = {}

    # -------------------------
    # Forward
    # -------------------------
    begin_memory_stage()

    with record_function("forward"):
        logits = model(tokens)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    stage_stats["after_forward"] = memory_snapshot()

    # -------------------------
    # Backward
    # -------------------------
    begin_memory_stage()

    with record_function("backward"):
        loss.backward()

    stage_stats["after_backward"] = memory_snapshot()

    # -------------------------
    # Optimizer
    # -------------------------
    begin_memory_stage()

    with record_function("optimizer"):
        optimizer.step()

    stage_stats["after_optimizer"] = memory_snapshot()

    return loss, stage_stats


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    output_dir = Path("results/memory")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("PyTorch:", torch.__version__)
    print("CUDA device:", torch.cuda.get_device_name(device))

    # 固定随机种子，保证实验可复现
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 实验配置
    vocab_size = 8192
    batch_size = 4
    context_length = 512
    d_model = 512
    nhead = 8
    num_layers = 4
    warmup_steps = 3

    print("\nConfiguration:")
    print("  batch_size:", batch_size)
    print("  context_length:", context_length)
    print("  vocab_size:", vocab_size)
    print("  d_model:", d_model)
    print("  nhead:", nhead)
    print("  num_layers:", num_layers)

    # 构造模型
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    # 随机 token 输入和目标
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )

    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )

    # 清理之前实验留下的缓存
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nWarmup...")

    # 预热，避免 CUDA 初始化和第一次内存分配影响结果
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)

        logits = model(tokens)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    print("Warmup finished.")
    print(
        "Memory after warmup:",
        json.dumps(memory_snapshot(), indent=2),
    )

    # 重新清空梯度，但不释放模型和 optimizer 状态
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    print("\nRunning torch.profiler...")

    # 使用 PyTorch profiler 记录 CUDA 时间和显存
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loss, stage_stats = run_profiled_train_step(
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            targets=targets,
        )
        prof.step()

    torch.cuda.synchronize()

    # 保存阶段显存统计
    stage_output_path = output_dir / "stage_memory.json"
    stage_output_path.write_text(
        json.dumps(
            stage_stats,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 保存总实验信息
    final_memory = memory_snapshot()

    experiment_result = {
        "experiment": "Transformer memory profiling",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "warmup_steps": warmup_steps,
        "loss": float(loss.detach().float().cpu()),
        "stage_memory": stage_stats,
        "final_memory": final_memory,
    }

    result_path = output_dir / "memory_profile.json"
    result_path.write_text(
        json.dumps(
            experiment_result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 按 CUDA 显存占用排序，导出 profiler 表格
    profiler_table = prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=40,
    )

    table_path = output_dir / "memory_profiler_table.txt"
    table_path.write_text(
        profiler_table,
        encoding="utf-8",
    )

    # 导出 Chrome/Perfetto 时间线
    trace_path = output_dir / "memory_trace.json"
    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception as exc:
        print("Could not export Chrome trace:", repr(exc))

    # 导出 memory timeline
    timeline_path = output_dir / "memory_timeline.html"
    try:
        prof.export_memory_timeline(str(timeline_path))
    except Exception as exc:
        print("Could not export memory timeline:", repr(exc))

    # 保存 CUDA 显存摘要
    summary_path = output_dir / "cuda_memory_summary.txt"
    summary_path.write_text(
        torch.cuda.memory_summary(),
        encoding="utf-8",
    )

    print("\n===== Stage Memory =====")
    print(json.dumps(stage_stats, indent=2))

    print("\n===== Final Memory =====")
    print(json.dumps(final_memory, indent=2))

    print("\nSaved files:")
    print(result_path)
    print(stage_output_path)
    print(table_path)
    print(trace_path)
    print(timeline_path)
    print(summary_path)


if __name__ == "__main__":
    main()