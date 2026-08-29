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
from torch.utils.checkpoint import checkpoint_sequential

def forward_with_checkpoint(model, inputs, checkpoint_block_size):

    x = model.token_embeddings(inputs)

    if checkpoint_block_size is None:
        for layer in model.layers:
            x = layer(x)

    else:
        num_segments = len(model.layers) // checkpoint_block_size
        x = checkpoint_sequential(model.layers, num_segments, x, use_reentrant=False)

    x = model.ln_final(x)
    logits = model.lm_head(x)

    return logits

def run_train_step(input_ids, model, targets, optimizer, vocab_size,run_autocast=False,checkpoint_block_size=None):
    optimizer.zero_grad()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if run_autocast else nullcontext()
    with autocast_ctx:
        logits = forward_with_checkpoint(model=model, inputs=input_ids, checkpoint_block_size=checkpoint_block_size)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()


    return logits,loss


def measurement(
        warmup,
        steps,
        input_ids, 
        model, 
        targets, 
        optimizer, 
        vocab_size,
        run_autocast,
        args,
        checkpoint_block_size
    ):
    results = {'step_time_ms_samples':'-', 'step_time_ms_p50':'-', 'peak_allocated_mib':'-', 'peak_reserved_mib':'-', 'status':'-'}

    try :
        for i in range(warmup):
            run_train_step(
                input_ids=input_ids,
                model=model, 
                targets=targets, 
                optimizer=optimizer, 
                vocab_size=vocab_size,
                run_autocast=run_autocast,
                checkpoint_block_size=checkpoint_block_size
            )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        timings = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for i in range(steps):
            start.record()
            logits, loss = run_train_step(
                    input_ids=input_ids,
                    model=model,
                    targets=targets,
                    optimizer=optimizer,
                    vocab_size=vocab_size,
                    run_autocast=run_autocast,
                    checkpoint_block_size=checkpoint_block_size
                )
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))

        peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024 ** 2)
        results['status'] = 'success'
        results['step_time_ms_samples'] = ",".join(map(str,timings))
        results['step_time_ms_p50'] = statistics.median(timings)
        results['peak_allocated_mib'] = peak_allocated_mib
        results['peak_reserved_mib'] = peak_reserved_mib


    except torch.cuda.OutOfMemoryError:
        results['status'] = 'oom'
        torch.cuda.empty_cache()
    
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        torch.cuda.empty_cache()
        
    except Exception as e :
        print(f"Exception:{e}")
        results['status'] = 'error'
        torch.cuda.empty_cache()


    return results


def main():
    #参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--dtype", type=str, required=True, choices=["FP32", "BF16", "FP16", "autocast"])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--warmup",type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--checkpoint-block-size", type=int, required=True)
    parser.add_argument("--config-id", type=int, required=True)

    args = parser.parse_args()

    d_model_dict = {"small":768, "medium":1024,"large":1280, "xl":2560, "10B":4608}
    num_layers_dict = {"small":12, "medium":24,"large":36, "xl":32, "10B":50}
    num_heads_dict =  {"small":12, "medium":16,"large":20, "xl":32, "10B":36}
    d_ff_dict = {"small":3072, "medium":4096,"large":5120, "xl":10240, "10B":12288}
    dtype_dict = {"FP32":torch.float32, "BF16":torch.bfloat16, "FP16":torch.float16}

    vocab_size = 10000
    batch_size = args.batch_size
    context_length = args.context_length
    seed = args.seed
    warmup = args.warmup
    steps = args.steps
    d_model = d_model_dict[args.model_size]
    num_layers = num_layers_dict[args.model_size]
    num_heads = num_heads_dict[args.model_size]
    d_ff = d_ff_dict[args.model_size]
    config_id=args.config_id


    run_autocast = False
    if args.dtype in dtype_dict:
        dtype = dtype_dict[args.dtype]
    else:
        dtype = torch.float32
        run_autocast = True

    checkpoint_block_size = None
    if args.checkpoint_block_size is not None and args.checkpoint_block_size > 0:
        checkpoint_block_size = args.checkpoint_block_size 
    


    
    if not torch.cuda.is_available():
        raise RuntimeError("This bench requires CUDA, but CUDA is not available")
    print("cuda is available")
    device = "cuda"

    if args.dtype == "BF16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16")
    
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 设置 23 GiB allocator 上限
    total = torch.cuda.get_device_properties(0).total_memory
    limit = 23 * 1024**3
    fraction = min(1.0, limit / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    
    print(f"Allocator limit: {limit / (1024**3):.2f} GiB, fraction: {fraction:.4f}")

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    model = model.to(device=device, dtype=dtype)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3, 
        weight_decay=0.01
    )

    input_ids = torch.randint(0, vocab_size,(batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size,(batch_size, context_length), device=device)

    results = measurement(
        warmup=warmup,
        steps=steps,
        input_ids=input_ids, 
        model=model, 
        targets=targets, 
        optimizer=optimizer, 
        vocab_size=vocab_size,
        run_autocast=run_autocast,
        args=args,
        checkpoint_block_size=checkpoint_block_size
    )

    csv_path = "local_results/checkpointing.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        fieldnames = ['config_id', 'model_size', 'num_layers', 'context_length', 'batch_size', 'dtype', 'checkpoint_block_size', 'nested', 'warmup_steps', 'measurement_steps', 'step_time_ms_samples', 'step_time_ms_p50', 'peak_allocated_mib', 'peak_reserved_mib', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "config_id":config_id,
            "model_size":args.model_size,
            "num_layers":num_layers,
            "context_length":context_length,
            "batch_size":batch_size,
            "dtype":args.dtype,
            "checkpoint_block_size":checkpoint_block_size if checkpoint_block_size is not None else "-",
            "nested": "No" if checkpoint_block_size is not None else "-",
            "warmup_steps" :warmup,
            "measurement_steps":steps,
            "step_time_ms_samples": results["step_time_ms_samples"],
            "step_time_ms_p50":results["step_time_ms_p50"],
            "peak_allocated_mib":results["peak_allocated_mib"],
            "peak_reserved_mib":results["peak_reserved_mib"],
            "status":results["status"]
            })

if __name__ == "__main__":
    main()