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
from nvtx_ranges import nvtx_range
from contextlib import nullcontext
from memory_snapshot import start_recording, dump_snapshot, stop_recording


def run_forward(input_ids, model, targets=None, optimizer=None, vocab_size=None,run_autocast=False, snapshot_callback=None, snapshot_filepath=None):

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if run_autocast else nullcontext()

    with autocast_ctx:
        with nvtx_range("forward"):
            with torch.no_grad():
                logits = model(input_ids)
                if snapshot_callback is not None:
                    full_path = f"{snapshot_filepath}_forward.pickle"
                    dump_snapshot(full_path)
    
    loss = None
    return logits, loss

def run_forward_backward(input_ids, model, targets, optimizer, vocab_size,run_autocast=False, snapshot_callback=None,snapshot_filepath=None):
    optimizer.zero_grad()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if run_autocast else nullcontext()

    with autocast_ctx:
        with nvtx_range("forward"):
            logits = model(input_ids)
            loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            if snapshot_callback is not None:
                full_path = f"{snapshot_filepath}_forward.pickle"
                dump_snapshot(full_path)

    with nvtx_range("backward"):
        loss.backward()
        if snapshot_callback is not None:
            full_path = f"{snapshot_filepath}_backward.pickle"
            dump_snapshot(full_path)

    return logits, loss

def run_train_step(input_ids, model, targets, optimizer, vocab_size,run_autocast=False, snapshot_callback=None,snapshot_filepath=None):
    optimizer.zero_grad()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if run_autocast else nullcontext()
    with autocast_ctx:
        with nvtx_range("forward"):
            logits = model(input_ids)
            loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            if snapshot_callback is not None:
                full_path = f"{snapshot_filepath}_forward.pickle"
                dump_snapshot(full_path)

    with nvtx_range("backward"):    
        loss.backward()
        if snapshot_callback is not None:
            full_path = f"{snapshot_filepath}_backward.pickle"
            dump_snapshot(full_path)

    with nvtx_range("optimizer"): 
        optimizer.step()
        if snapshot_callback is not None:
            full_path = f"{snapshot_filepath}_optimizer.pickle"
            dump_snapshot(full_path)

    return logits,loss

def measurement(mode, warmup, steps, input_ids, model, targets, optimizer, vocab_size, args, run_autocast=False, snapshot_callback=None):
    func_dict = {"forward":run_forward, "forward_backward":run_forward_backward, "train_step":run_train_step}
    func_measurement = func_dict[mode]

    torch.cuda.reset_peak_memory_stats()
    with nvtx_range("profile/warmup"):
        for i in range(warmup):
            func_measurement(
                input_ids=input_ids,
                model=model,
                targets=targets,
                optimizer=optimizer,
                vocab_size=vocab_size,
                run_autocast=run_autocast
            )

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    logits = 0.0
    loss = 0.0
    snapshot_filepath=None

    if snapshot_callback is not None:
        snapshot_filepath = f"results/memory/{args.model_size}_{mode}_{args.context_length}_after" 
        start_recording()


    with nvtx_range("profile/measure"):
        for i in range(steps):
        
            start.record()
            logits, loss = func_measurement(
                input_ids=input_ids,
                model=model,
                targets=targets,
                optimizer=optimizer,
                vocab_size=vocab_size,
                run_autocast=run_autocast,
                snapshot_filepath=snapshot_filepath,
                snapshot_callback=snapshot_callback
            )
            end.record()
            torch.cuda.synchronize()

            elapsed = start.elapsed_time(end)
            timings.append(elapsed)
    
    if snapshot_callback is not None:
        stop_recording()

    peak_mb = torch.cuda.max_memory_allocated()/(1024**2)  
    return timings, peak_mb, logits, loss


def make_hook(name, dtype_info):
    def hook(model, input, output):
        if isinstance(output, torch.Tensor):
            dtype_info[name] = str(output.dtype)
    return hook


def main():
    #参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", type=str, required=True, choices=["forward", "forward_backward","train_step"])
    parser.add_argument("--dtype", type=str, required=True, choices=["FP32", "BF16", "FP16", "autocast"])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--warmup",type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--memory-snapshot", action="store_true", default=False)

    args = parser.parse_args()

    #参数解析
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
    mode = args.mode
    

    run_autocast = False
    if args.dtype in dtype_dict:
        dtype = dtype_dict[args.dtype]
    else:
        dtype = torch.float32
        run_autocast = True
    
    if args.memory_snapshot:
        snapshot_callback = True
    else :
        snapshot_callback = None
    
    
    #设备设置
    if not torch.cuda.is_available():
        raise RuntimeError("This bench requires CUDA, but CUDA is not available")
    print("cuda is available")
    device = "cuda"
    
    if args.dtype == "BF16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16")
    

    #随机种子设置
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

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

    dtype_info = {}
    model.layers[0].register_forward_hook(make_hook("first_block_output", dtype_info))
    model.layers[0].ln1.register_forward_hook(make_hook("first_layernorm_output", dtype_info))


    timings, peak_mb, logits, loss = measurement(
        mode=mode,
        warmup=warmup,
        steps=steps,
        input_ids=input_ids, 
        model=model, 
        targets=targets, 
        optimizer=optimizer, 
        vocab_size=vocab_size,
        run_autocast=run_autocast,
        args=args,
        snapshot_callback=snapshot_callback
    )

    val_mean = statistics.mean(timings)
    val_pstdev = statistics.pstdev(timings)
    print(f"均值：{val_mean}, 标准差：{val_pstdev}")

    if args.output is not None:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{args.model_size}_{mode}_{context_length}_{args.dtype}"
        if warmup == 0:
            base_name += "_nowarmup"
        csv_path = os.path.join(output_dir, f"{base_name}_timings.csv")
        json_path = os.path.join(output_dir, f"{base_name}_metadata.json")

        with open(csv_path, "w", newline="") as f:
            fieldnames = ["step", "time_ms"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, t in enumerate(timings):
                writer.writerow({"step":i, "time_ms":t})
        


        logits_dtype = str(logits.dtype)
        loss_dtype = str(loss.dtype) if loss is not None else "N/A"
        dtype = {
            "logits_dtype":logits_dtype,
            "loss_dtype":loss_dtype
        }
        dtype["param_dtype"] = str(next(model.parameters()).dtype)
        dtype["first_block_output_dtype"] = dtype_info.get("first_block_output", "N/A")
        dtype["first_layernorm_output_dtype"] = dtype_info.get("first_layernorm_output", "N/A")
        for p in model.parameters():
            if p.grad is not None:
                dtype["grad_dtype"] = str(p.grad.dtype)
                break

        command = " ".join(sys.argv)
        metadata = {
        "command": "uv run --no-sync python " + command,
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": mode,
        "dtype": args.dtype,
        "warmup": warmup,
        "steps": steps,
        "mean_ms": val_mean,
        "std_ms": val_pstdev,
        "peak_mb": peak_mb,
        "dtype":dtype
    }

        with open(json_path, "w") as f :
            json.dump(metadata, f, indent=2)

        print(f"timings->{csv_path}")
        print(f"metadata->{json_path}")

if __name__ == "__main__":
    main()