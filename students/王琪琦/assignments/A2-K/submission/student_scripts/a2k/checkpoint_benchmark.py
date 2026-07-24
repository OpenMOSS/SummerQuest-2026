from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import TransformerLM


MODEL = dict(vocab_size=10_000, d_model=1024, num_layers=24, num_heads=16, d_ff=4096, rope_theta=10_000.0)
LIMIT_MIB = 23 * 1024


def guarded_device():
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, LIMIT_MIB * 2**20 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return torch.device("cuda"), fraction


def checkpointed_forward(model, tokens, block_size):
    hidden = model.token_embeddings(tokens)
    positions = torch.arange(tokens.shape[-1], device=tokens.device)
    for start in range(0, len(model.layers), block_size):
        layers = model.layers[start : start + block_size]
        def run_group(value, layers=layers):
            for layer in layers: value = layer(value, positions)
            return value
        hidden = checkpoint(run_group, hidden, use_reentrant=False)
    return model.lm_head(model.ln_final(hidden))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--context",type=int,required=True); p.add_argument("--block-size",type=int,choices=(1,2,4,8)); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    device,fraction=guarded_device(); torch.manual_seed(2026)
    result={"config_id":f"medium_c{a.context}_b{a.block_size or 0}","model_size":"medium","num_layers":24,"context_length":a.context,"batch_size":1,"dtype":"bf16_autocast","checkpoint_block_size":a.block_size or "none","nested":False,"warmup_steps":3,"measurement_steps":5,"allocator_limit_mib":LIMIT_MIB,"allocator_fraction":fraction,"status":"success"}
    try:
        model=TransformerLM(context_length=a.context,device=device,**MODEL); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); tokens=torch.randint(0,10_000,(1,a.context),device=device); targets=torch.randint_like(tokens,high=10_000)
        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(tokens) if a.block_size is None else checkpointed_forward(model,tokens,a.block_size)
                loss=F.cross_entropy(logits.flatten(0,1),targets.flatten())
            loss.backward(); optimizer.step()
        for _ in range(3): step(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(); samples=[]
        for _ in range(5):
            torch.cuda.synchronize(); start=time.perf_counter(); step(); torch.cuda.synchronize(); samples.append((time.perf_counter()-start)*1000)
        result.update(step_time_ms_samples=samples,step_time_ms_p50=statistics.median(samples),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
        result.update(status="oom" if isinstance(exc,torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "failed",error_type=type(exc).__name__,step_time_ms_samples=[],step_time_ms_p50="",peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result))

if __name__=="__main__": main()
