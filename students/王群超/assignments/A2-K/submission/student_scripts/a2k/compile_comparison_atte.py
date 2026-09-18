import argparse
import torch
import statistics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW 
from cs336_basics.nn_utils import cross_entropy
import json
import csv
import os
import sys
from contextlib import nullcontext
from triton.testing import do_bench

def attention(Q, K, V, is_causal=True):
    _, seq_len, head_dim = Q.shape
    scale = head_dim ** (-0.5)

    scores = Q @ K.transpose(-2, -1) * scale
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out

def eager_forward(Q, K, V, compiled_attention=None):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    attention(Q, K, V) 
    end.record()
    torch.cuda.synchronize()
    cold_start_ms = start.elapsed_time(end)

    for _ in range(10):
        attention(Q, K, V)
    torch.cuda.synchronize()

    fn = lambda: attention(Q, K, V)
    Q.grad = K.grad = V.grad = None
    return cold_start_ms ,do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def eager_forward_backward(Q, K, V, compiled_attention=None):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    attention(Q, K, V).sum().backward(retain_graph=True)
    end.record()
    torch.cuda.synchronize()
    cold_start_ms = start.elapsed_time(end)

    for _ in range(10):
        attention(Q, K, V).sum().backward(retain_graph=True)
    torch.cuda.synchronize()

    fn = lambda: attention(Q, K, V).sum().backward(retain_graph=True) 
    Q.grad = K.grad = V.grad = None
    return cold_start_ms ,do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def compiled_forward(Q, K, V, compiled_attention):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    compiled_attention(Q, K, V) 
    end.record()
    torch.cuda.synchronize()
    cold_start_ms = start.elapsed_time(end)

    for _ in range(10):
        compiled_attention(Q, K, V)
    torch.cuda.synchronize()

    fn = lambda: compiled_attention(Q, K, V)
    Q.grad = K.grad = V.grad = None
    return cold_start_ms ,do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def compiled_forward_backward(Q, K, V, compiled_attention):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    compiled_attention(Q, K, V).sum().backward(retain_graph=True)
    end.record()
    torch.cuda.synchronize()
    cold_start_ms = start.elapsed_time(end)

    for _ in range(10):
        compiled_attention(Q, K, V).sum().backward(retain_graph=True)
    torch.cuda.synchronize()

    fn = lambda: compiled_attention(Q, K, V).sum().backward(retain_graph=True) 
    Q.grad = K.grad = V.grad = None
    return cold_start_ms ,do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def main():

    configs = [
    (512, 64), 
    (2048, 128),
    (8192, 128)
    ]

    phases = ["eager_forward","eager_forward_backward", "compiled_forward", "compiled_forward_backward"]
    measure_fns = {"eager_forward":eager_forward,"eager_forward_backward":eager_forward_backward, "compiled_forward":compiled_forward, "compiled_forward_backward":compiled_forward_backward}
    

    total = torch.cuda.get_device_properties(0).total_memory
    limit = 23 * 1024**3
    fraction = min(1.0, limit / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    
    print(f"Allocator limit: {limit / (1024**3):.2f} GiB, fraction: {fraction:.4f}")


    for config_id, (seq_len, head_dim) in enumerate(configs):
        for phase in phases:
            status = 'success'
            p20_ms = p50_ms = p80_ms = None
            peak_allocated = 0.0
            peak_reserved = 0.0
            cold_start_ms = 0.0

   
            measure_fn = measure_fns[phase]

            Q = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            K = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            V = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            torch._dynamo.reset() 
            compiled_attention = None
            if phase in ["compiled_forward", "compiled_forward_backward"]:
                compiled_attention = torch.compile(attention, mode="max-autotune")
            
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                cold_start_ms,(p20_ms, p50_ms, p80_ms) = measure_fn(Q, K, V, compiled_attention)
                peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  
                

            except torch.cuda.OutOfMemoryError:
                status = 'oom'
                torch.cuda.empty_cache()
                
    
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                import traceback
                traceback.print_exc()
                status = 'error'
                torch.cuda.empty_cache()
                
        
            except Exception as e :
                print(f"Exception:{e}")
                status= 'error'
                torch.cuda.empty_cache()
                


            csv_path = "local_results/compile_comparison.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            file_exists = os.path.exists(csv_path)

            with open(csv_path, "a", newline="") as f:
                fieldnames = ['config_id', 'component','phase', 'context_length','head_dim','batch_size', 'dtype','warmup_steps', 'measurement_steps', 'cold_start_ms','steady_state_p20_ms', 'steady_state_p50_ms','steady_state_p80_ms', 'peak_allocated_mib', 'peak_reserved_mib', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'config_id':str(config_id),
                    'component':"attention",
                    'phase': phase,
                    'context_length': str(seq_len), 
                    'head_dim':str(head_dim),
                    'batch_size':'1', 
                    'dtype':"BF16",
                    'warmup_steps': '100',
                    'measurement_steps':'300',
                    'cold_start_ms': cold_start_ms,
                    'steady_state_p20_ms':p20_ms, 
                    'steady_state_p50_ms':p50_ms,
                    'steady_state_p80_ms':p80_ms, 
                    'peak_allocated_mib':peak_allocated, 
                    'peak_reserved_mib':peak_reserved, 
                    'status':status
                    })

if __name__ == "__main__":
    main()