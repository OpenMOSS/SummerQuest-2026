import torch
import triton
import csv
import json
import os
import sys
import math
import subprocess
from triton.testing import do_bench
from tests.adapters import get_flashattention_autograd_function_pytorch, get_flashattention_autograd_function_triton

def eager_attention(Q, K, V, is_causal=True):
    """显式 PyTorch attention 基线，不调用 fused attention"""
    _, seq_len, head_dim = Q.shape
    scale = head_dim ** (-0.5)

    scores = Q @ K.transpose(-2, -1) * scale
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out

def run_nvidia_smi():
    cmd = ["nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
        "--format=csv,noheader",
    ]
    output = subprocess.check_output(cmd).decode().strip()

    parts = [p.strip() for p in output.split(',')]
    return {
        "name":parts[0],
        "total_memory_mib":int(parts[1].replace(" MiB", "")),
        "free_memory_mib":int(parts[2].replace(" MiB","")),
        "driver_version": parts[3],
        "power_limit_w": float(parts[4].replace(" W", "")),
        "p_state": parts[5]
    }
    


def run_metadata():
    gpu_info = run_nvidia_smi()

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    allocator_limit_bytes = 23 * 1024**3
    allocator_fraction = min(1.0, allocator_limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=0)


    meta = {
        "seed": 42,
        "command": "uv run --no-sync python student_scripts/a2k/flash_benchmark.py",
        "gpu": {
            "name": gpu_info["name"],
            "total_memory_gib": gpu_info["total_memory_mib"] / 1024,
            "free_memory_gib_before_run": gpu_info["free_memory_mib"] / 1024,
            "driver_version": gpu_info["driver_version"],
            "power_limit_w": gpu_info["power_limit_w"],
            "p_state": gpu_info["p_state"],
        },
        "software": {
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "triton_version": triton.__version__,
        },
        "settings": {
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
            "allocator_limit_mib": 23552,
            "allocator_fraction": allocator_fraction,
            "timer": "triton.testing.do_bench",
            "do_bench_warmup_ms": 100,
            "do_bench_rep_ms": 300,
            "do_bench_quantiles": [0.2, 0.5, 0.8],
        },
        "timestamp": "2026-07-27T12:00:00+08:00",
    }
    return meta

def make_forward_fn(impl, Q, K, V, is_causal):
    if impl == 'eager':
        return lambda: eager_attention(Q, K, V, is_causal)
    elif impl == 'compiled':
        compiled_attention = torch.compile(eager_attention, mode="default")
        return lambda: compiled_attention(Q, K, V, is_causal)
    elif impl == 'triton':
        triton_attention = get_flashattention_autograd_function_triton().apply
        return lambda: triton_attention(Q, K, V, is_causal) 
    else:
        raise ValueError(impl)

def make_backward_fn(impl, Q, K, V, is_causal, dO):
    if impl == 'eager':
        out = eager_attention(Q, K, V, is_causal)
    elif impl == 'compiled':
        compiled_attention = torch.compile(eager_attention, mode='default')
        out = compiled_attention(Q, K, V, is_causal)
    elif impl == 'triton':
        triton_attention = get_flashattention_autograd_function_triton().apply
        out = triton_attention(Q, K, V, is_causal)
    else :
        raise ValueError(impl)

    loss = (out * dO).sum()
    Q.grad = K.grad = V.grad = None
    return lambda: loss.backward(retain_graph=True)
    
def make_forward_backward_fn(impl, Q, K, V, is_causal, dO):
    if impl == 'eager':
        return lambda: eager_attention(Q, K, V, is_causal).backward(dO)
    elif impl == 'compiled':
        compiled_attention = torch.compile(eager_attention, mode='default')
        return lambda: compiled_attention(Q, K, V, is_causal).backward(dO)
    elif impl == 'triton':
        triton_attention = get_flashattention_autograd_function_triton().apply
        return lambda : triton_attention(Q, K, V, is_causal).backward(dO)
    else:
        raise ValueError(impl)
    
def get_tiling_metadata(impl):
    """从实现中提取 Triton tile 等元数据；eager/compiled 返回空"""
    if impl != "triton":
        return {}
    
    return {"q_tile_size": 32, "k_tile_size": 32}

def benchmark(config):

    torch.manual_seed(42)
    device = 'cuda'

    
    Q = torch.randn((1, config["seq_len"], config["head_dim"]), device=device, dtype=config["dtype"], requires_grad=True)
    K = torch.randn((1, config["seq_len"], config["head_dim"]), device=device, dtype=config["dtype"], requires_grad=True)
    V = torch.randn((1, config["seq_len"], config["head_dim"]), device=device, dtype=config["dtype"], requires_grad=True)
    dO = torch.randn_like(Q)

    if config["phase"] == "forward":
        fn = make_forward_fn(config["impl"], Q, K, V, config["is_causal"])
    elif config["phase"] == "backward":
        fn = make_backward_fn(config["impl"], Q, K, V, config["is_causal"], dO)
    else:
        fn = make_forward_backward_fn(config["impl"], Q, K, V, config["is_causal"], dO)
    
    status = "success"
    p20_ms = p50_ms = p80_ms = None
    peak_allocated_mib = peak_reserved_mib = 0.0

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        p20_ms, p50_ms, p80_ms = do_bench(
            fn,
            warmup=100,
            rep=300,
            quantiles=[0.2, 0.5, 0.8],
            return_mode="median",  # 实际只要 quantiles 返回值
        )

        peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024 ** 2)

    except torch.cuda.OutOfMemoryError:
        status = "oom"
        torch.cuda.empty_cache()

    except Exception as e:
        status = f"error:{type(e).__name__}"
        torch.cuda.empty_cache()
        
    result = {
    **config,
    "p20_ms": p20_ms,
    "p50_ms": p50_ms,
    "p80_ms": p80_ms,
    "peak_allocated_mib": peak_allocated_mib,
    "peak_reserved_mib": peak_reserved_mib,
    "status": status,
    }
    result.update(get_tiling_metadata(config["impl"]))

    return result
    
def main():
    total = torch.cuda.get_device_properties(0).total_memory
    limit = 23 * 1024 ** 3
    fraction = min(1.0, limit / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)

    seq_lens = [512, 2048, 8192, 16384]
    head_dims = [64, 128]
    impls = ["eager", "compiled", "triton"]
    phases = ["forward", "backward", "forward_backward"]

    results = []
    print("start benchmark")
    for seq_len in seq_lens:
        for head_dim in head_dims:
            for phase in phases:
                eager_p50 = None #用于计算speedup
                for impl in impls:
                    config = {
                        "impl": impl,
                        "phase": phase,
                        "batch_size": 1,
                        "seq_len": seq_len,
                        "head_dim": head_dim,
                        "dtype": torch.bfloat16,
                        "is_causal": True,
                        }

                    result = benchmark(config)

                    if impl == 'eager' and result['status'] == 'success':
                        eager_p50 = result['p50_ms']

                    if eager_p50 is not None and result['status'] == 'success':
                        result["speed_up"] =  eager_p50 / result['p50_ms']
                    else :
                        result["speed_up"] = None
                    results.append(result)

    print("finish loop")
    result_path = "local_results/flash_benchmark.csv"
    os.makedirs("local_results", exist_ok=True)
    with open(result_path, 'w', newline="") as f:
        fieldnames = ["impl", "phase", "batch_size", "seq_len", "head_dim", "dtype",
                      "is_causal", "p20_ms", "p50_ms", "p80_ms",
                      "peak_allocated_mib", "peak_reserved_mib",
                      "q_tile_size", "k_tile_size",
                      "speed_up", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    meta_path = "local_results/run_metadata.json"
    meta = run_metadata()
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("finish")
if __name__ == "__main__":
    main()
