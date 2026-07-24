from __future__ import annotations
import csv,json,shutil
from pathlib import Path

RAW=Path("local_results/a2k"); DEST=Path("../SummerQuest-2026/students/王琪琦/assignments/A2-K")

def rows(path):
    with path.open(encoding="utf-8") as f:return list(csv.DictReader(f))
def write_csv(path,data):
    fields=[]
    for row in data: fields.extend(k for k in row if k not in fields)
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def chart(path,title,data,ylabel):
    maximum=max(v for _,v in data)*1.15; bars=[]
    for i,(label,value) in enumerate(data):
        x=70+i*100;h=220*value/maximum;y=295-h
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="65" height="{h:.1f}" fill="#087e8b"/><text x="{x+32}" y="320" text-anchor="middle" font-size="11">{label}</text><text x="{x+32}" y="{y-7:.1f}" text-anchor="middle" font-size="11">{value:.2f}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="350" viewBox="0 0 900 350"><rect width="100%" height="100%" fill="#f6f1e7"/><text x="25" y="32" font-size="22" font-family="serif">{title}</text><text x="25" y="55" font-size="12">{ylabel}</text><line x1="50" y1="295" x2="875" y2="295" stroke="#263238"/>{"".join(bars)}</svg>',encoding="utf-8")

def main():
    result=DEST/"results";assets=DEST/"assets";result.mkdir(exist_ok=True);assets.mkdir(exist_ok=True)
    shutil.copyfile(RAW/"correctness.json",result/"correctness.json");shutil.copyfile(RAW/"attention_baseline.csv",result/"attention_baseline.csv");shutil.copyfile(RAW/"flash_benchmark.csv",result/"flash_benchmark.csv");shutil.copyfile(RAW/"run_metadata.json",result/"run_metadata.json")
    test=(RAW/"unit_tests.txt").read_text(); root=str(Path.cwd()); test=test.replace(root,"<workspace>").replace(str(Path.home()),"<home>");(result/"unit_tests.txt").write_text(test)
    checkpoints=[json.loads(p.read_text()) for p in sorted(RAW.glob("checkpoint_c*.json"))];write_csv(result/"checkpointing.csv",[{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in checkpoints])
    compile_rows=[]
    for row in rows(RAW/"compile_comparison.csv"): compile_rows.append({"workload":"attention",**row})
    compile_rows.extend(rows(RAW/"model_compile.csv"));write_csv(result/"compile_comparison.csv",compile_rows)
    all_rows=checkpoints+rows(RAW/"attention_baseline.csv")+rows(RAW/"flash_benchmark.csv")+rows(RAW/"compile_comparison.csv")+rows(RAW/"model_compile.csv")
    def values(key):
        out=[]
        for row in all_rows:
            try: out.append(float(row.get(key,"")))
            except (TypeError,ValueError):pass
        return out
    metadata=json.loads((RAW/"run_metadata.json").read_text()); peak_a=max(values("peak_allocated_mib"));peak_r=max(values("peak_reserved_mib"))
    evidence={"allocator":{"allocator_fraction":metadata["allocator_fraction"],"allocator_limit_mib":23552},"hard_limit_mib":24576,"pytorch_peak_allocated_mib":peak_a,"pytorch_peak_reserved_mib":peak_r,"within_24gib":peak_r<=23552,"physical_gpu_note":"development GPU has 48 GiB; PyTorch allocator was capped at 23 GiB"};(result/"memory_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    flash=rows(RAW/"flash_benchmark.csv");selected=[r for r in flash if r["phase"]=="forward" and r["head_dim"]=="64" and r["sequence_length"] in {"512","2048","8192"}]
    chart(assets/"flash_forward_latency.svg","FlashAttention forward latency",[(f'{r["implementation"][:3]}-{r["sequence_length"]}',float(r["p50_ms"])) for r in selected if r["status"]=="success"],"p50 latency (ms), head_dim=64")
    ck=[r for r in checkpoints if r["context_length"]==1024];chart(assets/"checkpoint_tradeoff.svg","Checkpoint memory tradeoff",[(f'b{r["checkpoint_block_size"]}',float(r["peak_allocated_mib"])/1024) for r in ck],"peak allocated (GiB), context=1024")
if __name__=="__main__":main()
