from __future__ import annotations

import argparse
import csv
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile
from profiling.common import MODELS, annotate_attention, build_model, environment, step, sync, write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-size",choices=MODELS,required=True); p.add_argument("--context-length",type=int,required=True); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--warmup",type=int,default=5); p.add_argument("--output",required=True); p.add_argument("--trace",required=True); p.add_argument("--table",required=True); a=p.parse_args()
    d=torch.device("cuda"); torch.manual_seed(2026); m=build_model(a.model_size,a.context_length,d); annotate_attention(m); o=torch.optim.AdamW(m.parameters(),lr=1e-4); x=torch.randint(0,10000,(a.batch_size,a.context_length),device=d); y=torch.randint_like(x,high=10000)
    with torch.profiler.record_function("profile/warmup"):
        for _ in range(a.warmup): step(m,o,x,y,"train_step","fp32",d); sync(d)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
        with torch.profiler.record_function("profile/warmup"):
            pass
        with torch.profiler.record_function("profile/measure"): step(m,o,x,y,"train_step","fp32",d); sync(d)
    Path(a.trace).parent.mkdir(parents=True,exist_ok=True); prof.export_chrome_trace(a.trace)
    rows=[]
    for e in prof.key_averages():
        rows.append({"model":a.model_size,"context":a.context_length,"mode":"train_step","name":e.key,"calls":e.count,"cpu_total_us":e.cpu_time_total,"cuda_total_us":getattr(e,"device_time_total",0.0)})
    rows=sorted(rows,key=lambda r:max(r["cpu_total_us"],r["cuda_total_us"]),reverse=True)
    Path(a.table).parent.mkdir(parents=True,exist_ok=True)
    with open(a.table,"w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    write_json(a.output,{"status":"success","config":{"model_size":a.model_size,"context_length":a.context_length,"batch_size":a.batch_size,"warmup":a.warmup,"mode":"train_step","dtype":"fp32","tool":"torch.profiler","trace_file":Path(a.trace).name},"environment":environment(d),"events":rows})
if __name__ == "__main__": main()
