from __future__ import annotations
import csv, json, statistics, time
from pathlib import Path
import argparse, torch
import torch.nn.functional as F
from cs336_basics.model import TransformerLM

CONFIG=dict(vocab_size=10_000,context_length=512,d_model=768,num_layers=12,num_heads=12,d_ff=3072,rope_theta=10_000.0)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    total=torch.cuda.get_device_properties(0).total_memory; fraction=min(1.0,23*2**30/total); torch.cuda.set_per_process_memory_fraction(fraction,0); d=torch.device("cuda"); torch.manual_seed(2026)
    rows=[]
    for impl in ("eager","compiled"):
        for phase in ("forward","forward_backward","train_step"):
            model=TransformerLM(device=d,**CONFIG); fn=model if impl=="eager" else torch.compile(model,fullgraph=False); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); x=torch.randint(0,10_000,(1,512),device=d); y=torch.randint_like(x,high=10_000)
            def step():
                if phase=="train_step": optimizer.zero_grad(set_to_none=True)
                elif phase=="forward_backward": model.zero_grad(set_to_none=True)
                grad=torch.no_grad() if phase=="forward" else torch.enable_grad()
                with grad,torch.autocast("cuda",dtype=torch.bfloat16): out=fn(x); loss=None if phase=="forward" else F.cross_entropy(out.flatten(0,1),y.flatten())
                if loss is not None: loss.backward()
                if phase=="train_step": optimizer.step()
            torch.cuda.synchronize(); start=time.perf_counter(); step(); torch.cuda.synchronize(); cold=(time.perf_counter()-start)*1000
            for _ in range(3): step(); torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(); samples=[]
            for _ in range(5): torch.cuda.synchronize(); start=time.perf_counter(); step(); torch.cuda.synchronize(); samples.append((time.perf_counter()-start)*1000)
            rows.append({"workload":"small_transformer","implementation":impl,"sequence_length":512,"head_dim":"","phase":phase,"dtype":"bf16_autocast","cold_start_ms":cold,"steady_samples_ms":json.dumps(samples),"p50_ms":statistics.median(samples),"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20,"status":"success"})
            del model,fn,optimizer,x,y; torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__": main()
