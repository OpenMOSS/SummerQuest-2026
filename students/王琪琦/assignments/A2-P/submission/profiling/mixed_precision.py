from __future__ import annotations
import argparse, time
import torch
from torch import nn
from profiling.benchmark import run
from profiling.common import sync, write_json

class ToyModel(nn.Module):
    def __init__(self): super().__init__(); self.fc1=nn.Linear(32,10); self.ln=nn.LayerNorm(10); self.fc2=nn.Linear(10,7)
    def forward(self,x): return self.fc2(self.ln(torch.relu(self.fc1(x))))

def accumulation():
    out={}
    for name,ad,vd in (("fp32_fp32",torch.float32,torch.float32),("fp16_fp16",torch.float16,torch.float16),("fp32_fp16",torch.float32,torch.float16),("fp32_cast_fp16",torch.float32,torch.float16)):
        acc=torch.tensor(0.,dtype=ad)
        for _ in range(1000): acc += torch.tensor(.01,dtype=vd).float() if name=="fp32_cast_fp16" else torch.tensor(.01,dtype=vd)
        out[name]=float(acc)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); a=p.parse_args(); d=torch.device("cuda"); m=ToyModel().to(d); x=torch.randn(64,32,device=d); captures={}
    for name,module in (("first_layer",m.fc1),("layernorm",m.ln),("logits",m.fc2)): module.register_forward_hook(lambda _m,_i,o,n=name: captures.__setitem__(n,str(o.dtype)))
    with torch.autocast("cuda",dtype=torch.bfloat16): logits=m(x); loss=logits.float().square().mean()
    loss.backward(); toy={"parameters":str(next(m.parameters()).dtype),**captures,"loss":str(loss.dtype),"gradients":sorted({str(p.grad.dtype) for p in m.parameters()})}
    benches=[]
    for dtype in ("fp32","bf16"):
        ns=argparse.Namespace(model_size="small",batch_size=4,context_length=512,mode="forward_backward",warmup=5,steps=10,dtype=dtype,seed=2026,device="cuda",output=f"mixed_{dtype}.json"); benches.append(run(ns)); torch.cuda.empty_cache()
    write_json(a.output,{"status":"success","accumulation":accumulation(),"toy_model":toy,"benchmarks":benches})
if __name__ == "__main__": main()
