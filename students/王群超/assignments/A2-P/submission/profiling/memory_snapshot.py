import torch
import os

def start_recording(max_entries=1000000):
     torch.cuda.memory._record_memory_history(enabled="all",max_entries=max_entries)

def dump_snapshot(filepath):
    dirname = os.path.dirname(filepath)
    if dirname: 
        os.makedirs(dirname, exist_ok=True)
    torch.cuda.memory._dump_snapshot(filepath)

def stop_recording():
    torch.cuda.memory._record_memory_history(enabled=None)