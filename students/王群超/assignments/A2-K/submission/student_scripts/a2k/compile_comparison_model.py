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

def forward(model, input_ids, targets=None, optimizer=None, vocab_size=None):
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with autocast_ctx:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        model(input_ids)
        end.record()
        torch.cuda.synchronize()
        cold_start_ms = start.elapsed_time(end)
        
        for _ in range(10):
            model(input_ids)
        torch.cuda.synchronize()
    
    fn = lambda: model(input_ids)
    return cold_start_ms, do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def forward_backward(model, input_ids, targets, vocab_size, optimizer=None):
    # 先跑一次建立计算图
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with autocast_ctx:
        logits = model(input_ids)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        logits = model(input_ids)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        cold_start_ms = start.elapsed_time(end)
        
        for _ in range(10):
            logits = model(input_ids)
            loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
        torch.cuda.synchronize()
    
    def fn():
        logits = model(input_ids)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
    return cold_start_ms, do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])

def train_step(model, input_ids, targets, optimizer, vocab_size):
    def fn():
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
    
    # cold-start
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with autocast_ctx:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        cold_start_ms = start.elapsed_time(end)
        
        # steady-state
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    
    return cold_start_ms, do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])


def main():

    phases = ["eager_forward","eager_forward_backward", "eager_train_step", "compiled_forward", "compiled_forward_backward", "compiled_train_step"]
    measure_fns = {"eager_forward":forward,"eager_forward_backward":forward_backward, "eager_train_step":train_step ,"compiled_forward":forward, "compiled_forward_backward":forward_backward, "compiled_train_step":train_step}
    

    total = torch.cuda.get_device_properties(0).total_memory
    limit = 23 * 1024**3
    fraction = min(1.0, limit / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    
    print(f"Allocator limit: {limit / (1024**3):.2f} GiB, fraction: {fraction:.4f}")

    
    d_model = 768
    num_layers = 12
    num_heads = 12
    d_ff = 3072
    dtype = torch.bfloat16
    vocab_size = 10000
    context_length = 512
    batch_size = 1
    device = "cuda"
    input_ids = torch.randint(0, vocab_size,(batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size,(batch_size, context_length), device=device)

    model = BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    )
    model = model.to(device=device, dtype=dtype)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    config_id = 3
   
    for phase in phases:
        status = 'success'
        p20_ms = p50_ms = p80_ms = None
        peak_allocated = 0.0
        peak_reserved = 0.0
        cold_start_ms = 0.0
        
   
        measure_fn = measure_fns[phase]

        torch._dynamo.reset() 

        if phase in ["compiled_forward", "compiled_forward_backward", "compiled_train_step"]:
           
            compiled_model = torch.compile(model, mode="max-autotune")
            current_model = compiled_model
        else:
            current_model = model
            
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            cold_start_ms,(p20_ms, p50_ms, p80_ms) = measure_fn(model=current_model, input_ids=input_ids, targets=targets, optimizer=optimizer, vocab_size=vocab_size)
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
            fieldnames = ['config_id', 'component','phase', 'context_length','head_dim','batch_size', 'dtype','warmup_steps', 'measurement_steps','cold_start_ms','steady_state_p20_ms', 'steady_state_p50_ms','steady_state_p80_ms', 'peak_allocated_mib', 'peak_reserved_mib', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'config_id':str(config_id),
                'component':"transformer",
                'phase': phase,
                'context_length': str(context_length), 
                'head_dim': '64',
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
        config_id += 1

if __name__ == "__main__":
    main()