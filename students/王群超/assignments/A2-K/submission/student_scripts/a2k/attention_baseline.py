import torch
import csv
import os
import sys
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

def measure_forward(Q, K, V):
    fn = lambda: attention(Q, K, V)
    Q.grad = K.grad = V.grad = None
    return do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def measure_backward(Q, K, V):
    out = attention(Q, K, V)
    loss = out.sum()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    fn = lambda: loss.backward(retain_graph=True)
    Q.grad = K.grad = V.grad = None
    return do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def measure_forward_backward(Q, K, V):
    fn = lambda: attention(Q, K, V).sum().backward(retain_graph=True) 
    Q.grad = K.grad = V.grad = None
    return do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def main():

    configs = [
    (512, 64), (512, 128),
    (2048, 64), (2048, 128),
    (8192, 64), (8192, 128),
    ]

    phases = ["forward", "backward", "forward_backward"]
    measure_fns = {"forward":measure_forward, "backward":measure_backward, "forward_backward":measure_forward_backward}
    

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

            measure_fn = measure_fns[phase]

            Q = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            K = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            V = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                p20_ms, p50_ms, p80_ms = measure_fn(Q, K, V)
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
                


            csv_path = "local_results/attention_baseline.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            file_exists = os.path.exists(csv_path)

            with open(csv_path, "a", newline="") as f:
                fieldnames = ['config_id', 'phase', 'context_length','head_dim','batch_size', 'dtype','warmup_steps', 'measurement_steps', 'step_time_ms_p20', 'step_time_ms_p50','step_time_ms_p80', 'peak_allocated_mib', 'peak_reserved_mib', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'config_id':str(config_id),
                    'phase': phase,
                    'context_length': str(seq_len), 
                    'head_dim':str(head_dim),
                    'batch_size':'1', 
                    'dtype':"BF16",
                    'warmup_steps': '100',
                    'measurement_steps':'300', 
                    'step_time_ms_p20':p20_ms, 
                    'step_time_ms_p50':p50_ms,
                    'step_time_ms_p80':p80_ms, 
                    'peak_allocated_mib':peak_allocated, 
                    'peak_reserved_mib':peak_reserved, 
                    'status':status
                    })

if __name__ == "__main__":
    main()