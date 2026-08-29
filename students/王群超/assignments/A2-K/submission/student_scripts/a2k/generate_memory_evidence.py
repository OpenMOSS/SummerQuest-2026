import json
import csv
import os

def find_max_from_csv(csv_path, field_name):
    max_val = 0.0
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row[field_name])
                if val > max_val:
                    max_val = val
            
            except (ValueError, TypeError):
                continue

    return max_val if max_val > 0 else None

def main():
    csv_files = [
        "local_results/checkpointing.csv",
        "local_results/attention_baseline.csv",
        "local_results/flash_benchmark.csv",
    ]
    
    peak_allocated = 0.0
    peak_reserved = 0.0

    for csv_file in csv_files:
        alloc = find_max_from_csv(csv_file, "peak_allocated_mib")
        reserv = find_max_from_csv(csv_file, "peak_reserved_mib")
        
        if alloc and alloc > peak_allocated:
            peak_allocated = alloc
        if reserv and reserv > peak_reserved:
            peak_reserved = reserv
    
    
    with open("local_results/run_metadata.json", 'r') as f:
        meta = json.load(f)
    
    allocator_fraction = meta["settings"]["allocator_fraction"]
    allocator_limit_mib = meta["settings"]["allocator_limit_mib"]
    

    within_24gib = peak_reserved <= allocator_limit_mib
    
    evidence = {
        "allocator": {
            "allocator_fraction": allocator_fraction,
            "allocator_limit_mib": allocator_limit_mib
        },
        "hard_limit_mib": 24576,
        "pytorch_peak_allocated_mib": peak_allocated,
        "pytorch_peak_reserved_mib": peak_reserved,
        "within_24gib": within_24gib
    }
    
    os.makedirs("local_results", exist_ok=True)
    with open("local_results/memory_evidence.json", 'w') as f:
        json.dump(evidence, f, indent=2)
    
    print(f"Generated memory_evidence.json:")
    print(f"  peak_allocated: {peak_allocated:.2f} MiB")
    print(f"  peak_reserved: {peak_reserved:.2f} MiB")
    print(f"  within_24gib: {within_24gib}")

if __name__ == "__main__":
    main()