from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import triton

from cs336_systems.a2k import PyTorchFlashAttention, TritonFlashAttention, explicit_attention


LIMIT_MIB = 23 * 1024


def configure() -> tuple[torch.device, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, LIMIT_MIB * 2**20 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda"), fraction


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        for row in rows[1:]:
            fieldnames.extend(key for key in row if key not in fieldnames)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def memory() -> tuple[float, float]:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20


def quantile_bench(fn) -> tuple[float, float, float]:
    values = triton.testing.do_bench(fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])
    return tuple(float(x) for x in values)


def implementation(name: str):
    if name == "eager": return explicit_attention
    if name == "compiled": return torch.compile(explicit_attention, fullgraph=True)
    if name == "triton": return lambda q, k, v, causal: TritonFlashAttention.apply(q, k, v, causal)
    raise ValueError(name)


def bench_one(name: str, seq: int, dim: int, phase: str, causal: bool = True) -> dict:
    device, _ = configure(); torch.manual_seed(2026)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    row = {"implementation": name, "batch_size": 1, "sequence_length": seq, "head_dim": dim,
           "dtype": "bfloat16", "causal": causal, "phase": phase, "warmup_ms": 100,
           "rep_ms": 300, "p20_ms": "", "p50_ms": "", "p80_ms": "",
           "peak_allocated_mib": "", "peak_reserved_mib": "", "speedup_vs_eager": "",
           "query_tile": 64 if name == "triton" else "", "key_tile": 64 if name == "triton" else "",
           "num_warps": 4 if name == "triton" else "", "num_stages": 2 if name == "triton" else "", "status": "success"}
    row["error_type"] = ""
    try:
        q = torch.randn(1, seq, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True); v = torch.randn_like(q, requires_grad=True)
        do = torch.randn_like(q); fn = implementation(name)
        cold_start = time.perf_counter(); output = fn(q, k, v, causal); torch.cuda.synchronize()
        row["cold_start_ms"] = (time.perf_counter() - cold_start) * 1000
        if phase == "forward": measured = lambda: fn(q, k, v, causal)
        elif phase == "backward":
            def measured():
                q.grad = k.grad = v.grad = None
                output.backward(do, retain_graph=True)
        else:
            def measured():
                q.grad = k.grad = v.grad = None
                fn(q, k, v, causal).backward(do)
        p20, p50, p80 = quantile_bench(measured)
        row.update(p20_ms=p20, p50_ms=p50, p80_ms=p80)
        row["peak_allocated_mib"], row["peak_reserved_mib"] = memory()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        row["status"] = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "failed"
        row["error_type"] = type(exc).__name__
        row["cold_start_ms"] = ""
        try: row["peak_allocated_mib"], row["peak_reserved_mib"] = memory()
        except Exception: pass
    return row


def correctness(output: Path) -> None:
    device, fraction = configure(); rows=[]
    for seed in (7, 19, 2026):
        for dim in (32, 64, 128):
            for causal in (False, True):
                dtype = torch.float32 if seed == 7 and dim == 32 else torch.bfloat16
                torch.manual_seed(seed); q=torch.randn(2, 137, dim, device=device, dtype=dtype, requires_grad=True)
                k=torch.randn_like(q, requires_grad=True); v=torch.randn_like(q, requires_grad=True); do=torch.randn_like(q)
                ref=explicit_attention(q,k,v,causal); ref.backward(do); ref_grads=[q.grad.clone(),k.grad.clone(),v.grad.clone()]
                q.grad=k.grad=v.grad=None; actual=TritonFlashAttention.apply(q,k,v,causal); lse=[t for t in actual.grad_fn.saved_tensors if t.shape==(2,137)][0]
                actual.backward(do); act_grads=[q.grad,k.grad,v.grad]
                scores=q.detach().float()@k.detach().float().transpose(-1,-2)/math.sqrt(dim)
                if causal: scores=scores.masked_fill(torch.arange(137,device=device)[:,None]<torch.arange(137,device=device)[None,:],-torch.inf)
                lse_ref=torch.logsumexp(scores,dim=-1)
                metrics={"output":(actual.detach(),ref.detach()),"lse":(lse,lse_ref),"dq":(act_grads[0],ref_grads[0]),"dk":(act_grads[1],ref_grads[1]),"dv":(act_grads[2],ref_grads[2])}
                errors={}
                passed=True
                for name,(a,b) in metrics.items():
                    diff=(a.float()-b.float()).abs(); errors[name+"_max_abs"]=float(diff.max()); errors[name+"_max_rel"]=float((diff/(b.float().abs()+1e-6)).max())
                    passed &= torch.allclose(a.float(),b.float(),rtol=2e-2,atol=2e-2)
                rows.append({"seed":seed,"shape":[2,137,dim],"dtype":str(dtype),"causal":causal,"rtol":0.02,"atol":0.02,"passed":bool(passed),**errors})
    write_json(output,{"allocator_fraction":fraction,"cases":rows,"passed":sum(r["passed"] for r in rows),"failed":sum(not r["passed"] for r in rows)})


def matrix(output: Path, kind: str) -> None:
    rows=[]
    if output.is_file():
        with output.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if kind == "attention": specs=[("eager",s,d,p) for s in (512,2048,8192) for d in (64,128) for p in ("forward","backward","forward_backward")]
    elif kind == "compile": specs=[(impl,s,d,p) for s,d in ((512,64),(2048,128),(8192,128)) for impl in ("eager","compiled") for p in ("forward","backward","forward_backward")]
    else:
        specs=[(impl,s,d,p) for s in (512,2048,8192) for d in (64,128) for impl in ("eager","compiled","triton") for p in ("forward","backward","forward_backward")]
        specs += [(impl,16384,d,p) for d in (64,128) for impl in ("eager","triton") for p in ("forward","backward","forward_backward")]
    for spec in specs:
        key=tuple(map(str,spec))
        existing={(str(r["implementation"]),str(r["sequence_length"]),str(r["head_dim"]),str(r["phase"])) for r in rows}
        if key in existing: continue
        row=bench_one(*spec); rows.append(row); write_csv(output,rows); print(json.dumps(row),flush=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    eager={(r["sequence_length"],r["head_dim"],r["phase"]):r for r in rows if r["implementation"]=="eager" and r["status"]=="success"}
    for row in rows:
        base=eager.get((row["sequence_length"],row["head_dim"],row["phase"]))
        if base and row["status"]=="success": row["speedup_vs_eager"]=float(base["p50_ms"])/float(row["p50_ms"])
    write_csv(output,rows)


def metadata(output: Path) -> None:
    device,fraction=configure(); props=torch.cuda.get_device_properties(device)
    query=["nvidia-smi","--query-gpu=memory.total,memory.free,driver_version,power.limit,pstate","--format=csv,noheader,nounits","--id=0"]
    values=subprocess.run(query,capture_output=True,text=True,check=True).stdout.strip().split(", ")
    write_json(output,{"starter_commit":"ca8bc81a59b70516f7ebb2da4808daade877c736","seed":2026,"gpu":props.name,"gpu_total_mib":round(props.total_memory/2**20,1),"start_memory_total_mib":float(values[0]),"start_memory_free_mib":float(values[1]),"driver":values[2],"power_limit_w":float(values[3]),"pstate":values[4],"python":platform.python_version(),"pytorch":torch.__version__,"cuda":torch.version.cuda,"triton":triton.__version__,"tf32":False,"allocator_limit_mib":LIMIT_MIB,"allocator_fraction":fraction,"benchmark":{"warmup_ms":100,"rep_ms":300,"quantiles":[0.2,0.5,0.8]},"commands":["python -m student_scripts.a2k.experiments correctness|attention|compile|flash|metadata --output <result>"]})


def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("metadata","correctness","attention","compile","flash")); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if a.command=="metadata": metadata(a.output)
    elif a.command=="correctness": correctness(a.output)
    else: matrix(a.output,a.command)

if __name__=="__main__": main()
