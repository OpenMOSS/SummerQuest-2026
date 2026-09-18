from __future__ import annotations
import csv, json, pickle
from pathlib import Path

RAW=Path("results/wangqiqi_a2p")
DEST=Path("../SummerQuest-2026/students/王琪琦/assignments/A2-P")

def load(path): return json.loads(path.read_text())
def csv_write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def svg(path,title,labels,values,unit):
    width=760; height=360; top=max(values)*1.15
    bars=[]
    for i,(label,value) in enumerate(zip(labels,values)):
        x=90+i*150; h=240*value/top; y=300-h
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="90" height="{h:.1f}" fill="#087e8b"/><text x="{x+45}" y="325" text-anchor="middle">{label}</text><text x="{x+45}" y="{y-8:.1f}" text-anchor="middle">{value:.1f}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f6f1e7"/><text x="30" y="35" font-size="22" font-family="serif">{title}</text><text x="30" y="62" font-size="13">unit: {unit}</text><line x1="55" y1="300" x2="730" y2="300" stroke="#263238"/>{''.join(bars)}</svg>',encoding="utf-8")

def memory_timeline(snapshot_path, output_path, title):
    snapshot=pickle.loads(snapshot_path.read_bytes()); events=snapshot["device_traces"][0]
    final=sum(block["size"] for segment in snapshot["segments"] for block in segment["blocks"] if block["state"]=="active_allocated")
    net=sum(event.get("size",0) * (1 if event["action"]=="alloc" else -1 if event["action"]=="free_completed" else 0) for event in events)
    active=final-net; values=[active]
    for event in events:
        if event["action"]=="alloc": active+=event["size"]
        elif event["action"]=="free_completed": active-=event["size"]
        values.append(active)
    stride=max(1,len(values)//650); sampled=values[::stride]
    if sampled[-1]!=values[-1]: sampled.append(values[-1])
    maximum=max(sampled); points=" ".join(f"{55+i*670/max(1,len(sampled)-1):.1f},{300-220*v/max(1,maximum):.1f}" for i,v in enumerate(sampled))
    output_path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360"><rect width="100%" height="100%" fill="#f6f1e7"/><text x="30" y="35" font-size="22" font-family="serif">{title}</text><text x="30" y="62" font-size="13">Active allocated memory; peak {maximum/2**20:.1f} MiB</text><line x1="55" y1="300" x2="725" y2="300" stroke="#263238"/><polyline fill="none" stroke="#d1495b" stroke-width="2" points="{points}"/><text x="55" y="325">allocator events</text></svg>',encoding="utf-8")

def main():
    bench=[]
    for name in ("forward","forward_backward","train_step_w5","train_step_w0"):
        d=load(RAW/f"benchmark/{name}.json"); c=d["config"]
        bench.append({"model":c["model_size"],"batch_size":c["batch_size"],"context":c["context_length"],"mode":c["mode"],"warmup":c["warmup"],"steps":c["steps"],"dtype":c["dtype"],"raw_timings_ms":json.dumps(d["raw_timings_ms"]),"mean_ms":d["mean_ms"],"sample_std_ms":d["sample_std_ms"],"cv":d["cv"],"peak_allocated_mib":d["peak_allocated_mib"]})
    csv_write(DEST/"results/benchmark.csv",bench)
    rows=[]; runs=[]
    for model in ("small","medium"):
        for ctx in (256,512,1024):
            d=load(RAW/f"profile/{model}_{ctx}.json"); runs.append(d["config"])
            for e in d["events"]: rows.append({"model":model,"context":ctx,"mode":"train_step","dtype":"fp32","tool":"torch.profiler","name":e["name"],"calls":e["calls"],"cpu_total_us":e["cpu_total_us"],"cuda_total_us":e["cuda_total_us"]})
    csv_write(DEST/"results/profile/trace_summary.csv",rows)
    (DEST/"results/profile/run_metadata.json").write_text(json.dumps({"environment":d["environment"],"runs":runs},indent=2)+"\n")
    mixed=load(RAW/"mixed_precision.json"); (DEST/"results/mixed_precision.json").write_text(json.dumps(mixed,indent=2)+"\n")
    peaks=[]; memruns=[]
    for ctx in (128,2048):
        for mode,filemode in (("forward","forward"),("train_step","train")):
            d=load(RAW/f"memory/xl_{ctx}_{filemode}.json"); memruns.append(d["config"])
            peaks.append({"model":"xl","context":ctx,"batch_size":1,"mode":mode,"dtype":"fp32","status":d["status"],"peak_allocated_mib":d.get("peak_allocated_mib"),"peak_reserved_mib":d.get("peak_reserved_mib"),"peak_active_mib":d.get("peak_active_mib","")})
    for model,ctx in (("xl",1024),("large",2048)):
        d=load(RAW/f"memory/{model}_{ctx}_train.json"); memruns.append(d["config"])
        peaks.append({"model":model,"context":ctx,"batch_size":1,"mode":"train_step","dtype":"fp32","status":d["status"],"peak_allocated_mib":d.get("peak_allocated_mib"),"peak_reserved_mib":d.get("peak_reserved_mib"),"peak_active_mib":d.get("peak_active_mib","")})
    csv_write(DEST/"results/memory/peaks.csv",peaks); (DEST/"results/memory/run_metadata.json").write_text(json.dumps({"environment":d["environment"],"runs":memruns,"snapshots":"retained locally; not submitted"},indent=2)+"\n")
    assets=DEST/"assets"; assets.mkdir(exist_ok=True)
    svg(assets/"benchmark_latency.svg","End-to-end latency",["forward","fwd+bwd","train w5","train w0"],[r["mean_ms"] for r in bench],"ms")
    mb=mixed["benchmarks"]; svg(assets/"mixed_precision.svg","FP32 vs BF16 autocast",["FP32 time","BF16 time","FP32 GiB","BF16 GiB"],[mb[0]["mean_ms"],mb[1]["mean_ms"],mb[0]["peak_allocated_mib"]/100,mb[1]["peak_allocated_mib"]/100],"ms; memory scaled by 100")
    svg(assets/"memory_peaks.svg","XL memory profile",["c128 fwd","c128 train","c2048 fwd","c2048 train"],[r["peak_allocated_mib"]/1024 for r in peaks],"GiB")
    memory_timeline(RAW/"memory/xl_128_forward.pickle",assets/"memory_timeline_ctx128.svg","XL context 128 forward timeline")
    memory_timeline(RAW/"memory/xl_2048_forward.pickle",assets/"memory_timeline_ctx2048.svg","XL context 2048 forward timeline")
    representative=[r for r in rows if r["model"]=="medium" and r["context"]==1024 and r["name"] in ("forward","backward","optimizer","aten::mm")]
    svg(assets/"compute_profile.svg","Medium context 1024 profile",[r["name"][:12] for r in representative[:4]],[r["cuda_total_us"]/1000 for r in representative[:4]],"CUDA ms")
if __name__=="__main__": main()
