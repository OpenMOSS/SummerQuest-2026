import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from env_utils import public_env
from nvtx_ranges import patch_attention

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

VOCAB_SIZE = 10000

PROFILE_DIR = Path("results/profile")


def run_key(model_size: str, batch_size: int, context_length: int, dtype: str) -> str:
    return f"{model_size}_ctx{context_length}_b{batch_size}_{dtype}"


def run_name(model_size: str, context_length: int) -> str:
    return f"run_{model_size}_ctx{context_length}"


def get_model(model_size, context_length, dtype):
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    )
    return model.to(dtype=dtype).cuda()


def write_failure(name: str, config: dict, stage: str, exc: BaseException) -> None:
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": name,  # matches the '<name>.nsys-rep' / '<name>_stats.csv' basenames
        "config": config,
        "stage": stage,  # one of: setup / warmup / measure
        "exception": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
        "exit_code": 1,
    }
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_DIR / "failures.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    config = {
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "steps": 1,  # 每次 nsys 运行只捕获一个测量 step
    }
    name = run_name(args.model_size, args.context_length)
    key = run_key(args.model_size, args.batch_size, args.context_length, args.dtype)

    # 每次 trace 都记录一份去敏环境元数据（命令 + 配置 + 版本），供报告溯源。
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    env_record = {"command": sys.argv, "config": config, "env": public_env()}
    (PROFILE_DIR / f"run_{key}_env.json").write_text(
        json.dumps(env_record, indent=2, default=str), encoding="utf-8"
    )

    device = torch.device("cuda")
    torch.manual_seed(42)
    stage = "setup"
    try:
        model = get_model(args.model_size, args.context_length, dtype)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)
        labels = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)

        # 预热步骤
        stage = "warmup"
        for _ in range(args.warmup):
            with nvtx.range("profile/warmup"):
                model.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss = cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()

        # 测量步骤
        stage = "measure"
        with nvtx.range("profile/measure"):
            with nvtx.range("forward"):
                model.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss = cross_entropy(logits, labels)
            with nvtx.range("backward"):
                loss.backward()
            with nvtx.range("optimizer"):
                optimizer.step()
            torch.cuda.synchronize()
    except Exception as exc:  
        write_failure(name, config, stage, exc)
        print(f"[profile_one_step] failed during '{stage}' for {name}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
