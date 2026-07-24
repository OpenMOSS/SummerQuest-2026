from __future__ import annotations
import argparse
from pathlib import Path
import torch
from profiling.common import MODELS, build_model, environment, step, sync, write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-size",choices=MODELS,default="xl"); p.add_argument("--context-length",type=int,required=True); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--mode",choices=("forward","train_step"),required=True); p.add_argument("--warmup",type=int,default=1); p.add_argument("--output",required=True); p.add_argument("--snapshot",required=True); a=p.parse_args(); d=torch.device("cuda"); result={"config":vars(a)|{"output":Path(a.output).name,"snapshot":Path(a.snapshot).name},"environment":environment(d)}
    try:
        m=build_model(a.model_size,a.context_length,d); o=torch.optim.AdamW(m.parameters(),lr=1e-4) if a.mode=="train_step" else None; x=torch.randint(0,10000,(a.batch_size,a.context_length),device=d); y=torch.randint_like(x,high=10000)
        for _ in range(a.warmup): step(m,o,x,y,"forward","fp32",d); sync(d)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(d); torch.cuda.memory._record_memory_history(max_entries=100000)
        step(m,o,x,y,a.mode,"fp32",d); sync(d); stats=torch.cuda.memory_stats(d); Path(a.snapshot).parent.mkdir(parents=True,exist_ok=True); torch.cuda.memory._dump_snapshot(a.snapshot)
        result.update(status="success",peak_allocated_mib=torch.cuda.max_memory_allocated(d)/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved(d)/2**20,peak_active_mib=stats["active_bytes.all.peak"]/2**20)
    except torch.cuda.OutOfMemoryError as e:
        result.update(status="oom",error_type=type(e).__name__,peak_allocated_mib=torch.cuda.max_memory_allocated(d)/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved(d)/2**20)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    write_json(a.output,result); print(result)
if __name__ == "__main__": main()
