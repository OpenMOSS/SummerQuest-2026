from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path) -> list[dict]:
    rows = []
    for file in sorted(root.rglob("*.json")):
        if "results" in file.parts or "smoke" in file.name:
            continue
        value = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "kind" in value:
            value["source"] = str(file.relative_to(root))
            rows.append(value)
    return rows


def number(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(number(cell) for cell in row) + " |" for row in rows)
    return out


def environment(rows: list[dict]) -> str:
    for row in rows:
        info = row.get("metadata", {})
        if info.get("device_name"):
            return f"{info['device_name']}; CUDA {info.get('cuda')}; PyTorch {info.get('torch')}; Triton {info.get('triton')}"
    return "not recorded"


def report_p(root: Path, rows: list[dict]) -> None:
    base = [x for x in rows if x["kind"] == "model_benchmark" and x.get("model_size") == "small" and x.get("dtype") == "float32"]
    profiles = [x for x in rows if x["kind"] == "profile"]
    memory = [x for x in rows if x["kind"] == "memory_profile"]
    mixed = [x for x in rows if x["kind"] == "mixed_precision"]
    lines = [
        "# A2-P：Profiling 与性能分析",
        "",
        "## 完成范围与环境",
        "",
        "本报告对应本地 `assignment2-systems` 工作目录和结果目录中的轻量 JSON/CSV。所有 A2-P 正式测量来自单张 RTX 3090；GPU、CUDA 和库版本如下。",
        "",
        f"`{environment(rows)}`",
        "",
        "每次被测 CUDA step 在开始和结束处同步；初始化和随机数据创建不计入步耗时。每个 benchmark JSON 保留 raw timing、样本标准差和 CV。",
        "",
        "## End-to-end benchmark",
        "",
    ]
    lines += table(["mode", "warm-up", "mean (s)", "sample std (s)", "CV", "status", "source"], [[x.get("mode"), x.get("warmup"), x.get("mean_seconds"), x.get("sample_stdev_seconds"), x.get("cv"), x.get("status"), x.get("source")] for x in base])
    lines += ["", "![Baseline timing](assets/benchmark_baseline.png)", "", "未 warm-up 的 train step 与稳态测量分开记录；首轮会额外包含 CUDA context、allocator 和 kernel 选择等冷启动开销，因此不能与充分 warm-up 的均值混为一谈。", "", "## Compute profiling", "", "共完成两个模型规模、三个 context length 的 6 个完整 train-step profile。每个 profile 包含 `profile/measure`、`forward`、`backward`、`optimizer` 与 attention 的 score / softmax / value 标记；完整 trace 未进入提交目录。"]
    lines += table(["model", "batch", "context", "top CUDA op", "CUDA total (us)", "source"], [[x.get("model_size"), x.get("batch_size"), x.get("context_length"), (x.get("top_ops") or [{}])[0].get("name"), (x.get("top_ops") or [{}])[0].get("cuda_total_us"), x.get("source")] for x in profiles])
    lines += ["", "## Mixed precision", "", "ToyModel 与固定累加实验保存在 `mixed/toy-and-accumulation.json`。BF16 autocast 保持参数和梯度为 FP32，而矩阵乘输出可为 BF16；LayerNorm / loss 等数值敏感归约保持 FP32。", "", "![Mixed precision](assets/mixed_precision.png)", "", "## Memory profiling", ""]
    lines += table(["model", "context", "mode", "status", "peak allocated (GiB)", "peak reserved (GiB)", "source"], [[x.get("model_size"), x.get("context_length"), x.get("mode"), x.get("status"), (x.get("memory") or {}).get("allocated_peak_bytes", 0) / 2**30, (x.get("memory") or {}).get("reserved_peak_bytes", 0) / 2**30, x.get("source")] for x in memory])
    lines += ["", "XL / context 2048 及 XL 的训练步在 24GB GPU 上发生 OOM，保留了失败行和峰值，而没有用缩小 shape 替换。为提供完整的阶段轨迹，另运行并明确标记了 medium / 512 / train-step fallback。", "", "![Peak memory](assets/memory_peak.png)", "", "![XL forward memory timeline](assets/memory_timeline_xl_l128_forward.png)", "", "![Fallback train-step memory timeline](assets/memory_timeline_medium_l512_train_step.png)", "", "## 可复现性与文件边界", "", "原始轻量结果位于 `results/`；图片位于 `assets/`。未提交 trace、snapshot、模型权重、数据集、缓存、内部路径、账号或凭据。"]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_k(root: Path, rows: list[dict]) -> None:
    ckpt = [x for x in rows if x["kind"] == "checkpoint"]
    attn = [x for x in rows if x["kind"] == "attention"]
    correct = [x for x in rows if x["kind"] == "flash_correctness"]
    lines = ["# A2-K：单卡显存优化与 GPU Kernels", "", "## 环境与偏离说明", "", "本作业原计划使用 RTX 4090。由于 4090 分区长期没有空闲卡，全部 A2-K 正式实验改在单张 RTX 3090 24GB 上完成。该硬件偏离会影响延迟和吞吐数值，但不改变 correctness、显存趋势和实现验证；所有表格和图均明确对应以下环境。", "", f"`{environment(rows)}`", "", "每个正式配置在独立 Python 进程中串行运行，并在首次 CUDA allocation 前设置 23552 MiB PyTorch allocator 上限。", "", "## Activation checkpointing", ""]
    lines += table(["context", "block size", "status", "mean step (s)", "peak allocated (GiB)", "source"], [[x.get("context_length"), x.get("block_size"), x.get("status"), x.get("mean_seconds"), (x.get("memory") or {}).get("allocated_peak_bytes", 0) / 2**30, x.get("source")] for x in ckpt])
    lines += ["", "Checkpointing does not store all block activations; backward recomputes each checkpointed region. Thus smaller saved-activation memory is exchanged for a longer train step.", "", "![Checkpoint trade-off](assets/checkpoint_tradeoff.png)", "", "## FlashAttention correctness", ""]
    if correct:
        values = correct[0].get("rows", [])
        lines += table(["implementation", "cases", "max O/LSE/dQ/dK/dV error", "status"], [[impl, len([x for x in values if x["implementation"] == impl]), max(max(x[k] for k in ("max_abs_o", "max_abs_lse", "max_abs_dq", "max_abs_dk", "max_abs_dv")) for x in values if x["implementation"] == impl), "pass" if all(x["status"] == "pass" for x in values if x["implementation"] == impl) else "fail"] for impl in sorted({x["implementation"] for x in values})])
    lines += ["", "The tiled PyTorch reference and Triton forward maintain online FP32 softmax state and save only LSE per query row. Gradients are validated against the explicit reference for three seeds, dimensions 32/64/128, and causal/non-causal modes.", "", "## Attention and compile performance", ""]
    core = [x for x in attn if x.get("sequence_length") in (512, 2048, 8192)]
    lines += table(["implementation", "sequence", "dim", "phase", "p50 (s)", "peak allocated (GiB)", "status"], [[x.get("implementation"), x.get("sequence_length"), x.get("dimension"), x.get("phase"), x.get("p50_seconds"), (x.get("memory") or {}).get("allocated_peak_bytes", 0) / 2**30, x.get("status")] for x in core])
    lines += ["", "![Attention latency](assets/attention_latency.png)", "", "Long-sequence rows and any OOM/compile failures remain in `results/` and are not silently removed. `unit_tests.txt` records the GPU unit-test result. No cache, trace, binary, model weight, internal host name or secret is included in the final submission."]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2p", required=True)
    parser.add_argument("--a2k", required=True)
    args = parser.parse_args()
    p_root, k_root = Path(args.a2p), Path(args.a2k)
    report_p(p_root, load(p_root))
    report_k(k_root, load(k_root))


if __name__ == "__main__":
    main()
