import os
import csv
import glob


def parse_filename(filepath):

    #取出纯文件名
    base = os.path.basename(filepath)
    #去掉拓展名
    name, ext = os.path.splitext(base)

    #去掉末尾标记后缀
    if name.endswith("_nvtx"):
        name = name[:-5]
    elif name.endswith("_cuda_gpu_kern_sum"):
        name = name[:-18]
    
    #按下划线拆分后中间重新拼接
    parts = name.split("_")

    model_size = parts[0]
    mode = "_".join(parts[1:-1])
    context_length = parts[-1]

    return model_size, mode, context_length

def parse_nvtx_file(filepath):
    result = {
        "forward": None,
        "backward": None,
        "optimizer": None,
        "profile/measure": None
    }

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "PushPop" not in line:
                continue
            if ":" not in line:
                continue

            cols = line.split()

            range_name = cols[-1]

            if range_name not in [":forward", ":backward", ":optimizer", ":profile/measure"]:
                continue
            
            #提取中位数
            med_str = cols[4].replace(",", "")
            med_ns = float(med_str)          # 纳秒
            med_ms = med_ns / 1_000_000.0    # 转成毫秒

            key = range_name[1:]
            result[key] = med_ms
    
    return result

def parse_kernel_csv(filepath):
    
    total_time_ns = 0.0
    gemm_time_ns = 0.0
    total_instance = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            time_ns = float(row["Total Time (ns)"].replace(",",""))
            instance = int(row["Instances"].replace(",",""))

            total_time_ns += time_ns
            total_instance += instance

            name = row["Name"]
            if "sgemm" in name:
                gemm_time_ns += time_ns
        
    
    gemm_percent = (gemm_time_ns / total_time_ns) * 100.0

    return gemm_percent, total_instance

def main():
    nsys_dir = "results/nsys"
   
    nvtx_files = glob.glob(os.path.join(nsys_dir, "*_nvtx.txt"))
    nvtx_data = []
   

    for f in nvtx_files:
        model_size, mode, context_length = parse_filename(f)
        data = parse_nvtx_file(f)
        nvtx_data.append((model_size, mode, context_length, data))

    print("\nNVTX Timing Summary (中位数, 单位: ms)")
    print("| model_size | context_length | mode | forward (ms) | backward (ms) | optimizer (ms) | measurement (ms)")
    print("|-------|---------|------|--------------|---------------|----------------|---------------|")

    for model_size, mode, context_length, data in nvtx_data:
        fwd = f"{data['forward']:.2f}" if data['forward'] is not None else "—"
        bwd = f"{data['backward']:.2f}" if data['backward'] is not None else "—"
        opt = f"{data['optimizer']:.2f}" if data['optimizer'] is not None else "—"
        mea = f"{data['profile/measure']:.2f}" if data['profile/measure'] is not None else "—"
        print(f"| {model_size} | {context_length} | {mode} | {fwd} | {bwd} | {opt} | {mea} |")

        
    kernel_files = glob.glob(os.path.join(nsys_dir, "*_cuda_gpu_kern_sum.csv"))
    kernel_data = []
    
    for f in kernel_files:
        model, mode, context = parse_filename(f)
        gemm_pct, total_inst = parse_kernel_csv(f)
        kernel_data.append((model, mode, context, gemm_pct, total_inst))
    
    
    print("\n CUDA Kernel Summary")
    print("| model | context | mode | GEMM % | Total Kernel Calls |")
    print("|-------|---------|------|--------|---------------------|")
    
    for model_size, mode, context_length, gemm_pct, total_inst in kernel_data:
        print(f"| {model_size} | {context_length} | {mode} | {gemm_pct:.1f}% | {total_inst:,} |")
    
if __name__ == "__main__":
    main()