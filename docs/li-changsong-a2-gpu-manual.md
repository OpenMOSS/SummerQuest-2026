# 李畅松 A2 GPU 实验逐步操作手册

这份文档只负责需要 GPU 的实验。命令默认两个仓库位于同一父目录：

```text
lichangsong-CZXS25250012/
├── SummerQuest-2026/
└── assignment2-systems/
```

不要把本文中的 `GPU_NOT_RUN` 当作结果。只有真实执行命令产生的数字才能写入报告。

## 0. 第一次运行：检查仓库和 GPU

```bash
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/assignment2-systems
git rev-parse HEAD
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,power.limit,pstate --format=csv,noheader
```

检查：

1. `git rev-parse HEAD` 应为或包含固定 starter `ca8bc81a59b70516f7ebb2da4808daade877c736`。
2. A2-K 正式实验必须使用单张 RTX 4090 24GB。
3. A2-K 开始时空闲显存必须不少于 22 GiB；不足时等待，不要缩小题目要求的 shape。

## 1. 使用已配置的 Conda 环境

本实验环境已经创建在：

```text
/inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/envs/summerquest-a2
```

当前安装的主要版本为 Python 3.13.14、PyTorch 2.11.0 和 Triton 3.6.0。A2 外层工程和
`cs336-basics` 均以 editable 模式安装，因此在 `assignment2-systems` 中修改 Python 源码后
不需要重复安装。

每次打开新终端，先执行：

```bash
source /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/etc/profile.d/conda.sh
conda activate /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/envs/summerquest-a2
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/assignment2-systems
```

确认没有误用其他环境，并检查依赖：

```bash
which python
python --version
python -c "import torch, triton, cs336_basics, cs336_systems; print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'triton': triton.__version__, 'cuda_available': torch.cuda.is_available()}); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
```

`which python` 必须指向上述 `summerquest-a2/bin/python`。在 GPU 节点上，
`cuda_available` 应为 `True`，最后一项应为实际 GPU 名称。若当前节点没有 GPU，环境导入仍可
验证，但不要生成或填写正式实验结果。

如果环境被删除或依赖损坏，可按下列顺序重建；不要在正常实验前反复执行：

```bash
CONDA=/inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/bin/conda
ENV=/inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/envs/summerquest-a2
$CONDA create -y -p "$ENV" python=3.13 pip
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/assignment2-systems
"$ENV/bin/python" -m pip install -e ./cs336-basics
"$ENV/bin/python" -m pip install -e .
```

后文命令默认已经激活该环境，因此统一使用 `python`。如果不想激活环境，也可以把每条命令
中的 `python` 替换为完整路径
`/inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/envs/summerquest-a2/bin/python`。

# A2-P：逐步运行

## 2. 建立本地结果目录

```bash
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/assignment2-systems
mkdir -p results/a2p/benchmark results/a2p/profile results/a2p/mixed results/a2p/memory
```

## 3. 运行 End-to-End Benchmark

基线要求 small、batch 4、context 512、FP32。逐条执行：

```bash
python -m profiling.benchmark --model-size small --batch-size 4 --context-length 512 --mode forward --dtype fp32 --warmup 5 --steps 10 --output results/a2p/benchmark/forward_warm5.json

python -m profiling.benchmark --model-size small --batch-size 4 --context-length 512 --mode forward_backward --dtype fp32 --warmup 5 --steps 10 --output results/a2p/benchmark/forward_backward_warm5.json

python -m profiling.benchmark --model-size small --batch-size 4 --context-length 512 --mode train_step --dtype fp32 --warmup 5 --steps 10 --output results/a2p/benchmark/train_step_warm5.json

python -m profiling.benchmark --model-size small --batch-size 4 --context-length 512 --mode train_step --dtype fp32 --warmup 0 --steps 10 --output results/a2p/benchmark/train_step_warm0.json
```

每个 JSON 必须包含 `timings_ms`、`mean_ms`、`std_ms` 和 `cv`。查看汇总：

```bash
python -m profiling.summarize results/a2p/benchmark/*.json
```

若出现 CUDA OOM，先运行 `nvidia-smi` 检查其他进程；不要修改规定的基线配置。

## 4. 运行六组 Compute Profile

使用 small/medium 两个模型和 256/512/1024 三个 context，共六组。每次只保留短 trace：

```bash
for model in small medium; do
  for ctx in 256 512 1024; do
    python -m profiling.torch_profile \
      --model-size "$model" --batch-size 1 --context-length "$ctx" \
      --mode train_step --dtype bf16 --warmup 5 --steps 1 \
      --output "results/a2p/profile/${model}_${ctx}.json" \
      --trace "results/a2p/profile/${model}_${ctx}_trace.json"
  done
done
```

检查文件：

```bash
ls -lh results/a2p/profile
```

在 Perfetto 打开一个代表性 `*_trace.json`，截取 forward、backward、optimizer 和 attention 关键区间。完整 trace 不复制到 SummerQuest，只保留脱敏截图及轻量汇总。

注意：当前 profiler 包装脚本会覆盖整个 `run()`。运行后需要确认 trace 中确实存在 CUDA activity；若没有，检查 `torch.cuda.is_available()` 和 CUDA 版 PyTorch。

## 5. Mixed Precision

先运行四种累加实验：

```bash
python -m profiling.mixed_precision --device cuda --n 1000 --value 0.1 --output results/a2p/mixed/accumulation.json
cat results/a2p/mixed/accumulation.json
```

再用相同配置分别跑 FP32/BF16 benchmark：

```bash
python -m profiling.benchmark --model-size small --batch-size 4 --context-length 512 --mode train_step --dtype fp32 --warmup 5 --steps 10 --output results/a2p/mixed/train_fp32.json
python -m profiling.benchmark --model-size small --batch-size 4 --context-length 512 --mode train_step --dtype bf16 --warmup 5 --steps 10 --output results/a2p/mixed/train_bf16.json
```

当前 `mixed_precision.py` 只覆盖四种累加实验；题面要求的 ToyModel dtype 跟踪仍需在上游工作区补脚本，记录参数、第一层输出、LayerNorm、logits、loss、gradient 的 dtype，以及 FP32/BF16 峰值显存。

## 6. Memory Snapshot

每个配置单独执行，避免显存状态相互污染：

```bash
python -m profiling.memory_snapshot --model-size xl --batch-size 1 --context-length 128 --mode forward --dtype fp32 --warmup 5 --steps 1 --output results/a2p/memory/xl_128_forward.json --snapshot results/a2p/memory/xl_128_forward.pickle

python -m profiling.memory_snapshot --model-size xl --batch-size 1 --context-length 128 --mode train_step --dtype fp32 --warmup 5 --steps 1 --output results/a2p/memory/xl_128_train.json --snapshot results/a2p/memory/xl_128_train.pickle

python -m profiling.memory_snapshot --model-size xl --batch-size 1 --context-length 2048 --mode forward --dtype fp32 --warmup 5 --steps 1 --output results/a2p/memory/xl_2048_forward.json --snapshot results/a2p/memory/xl_2048_forward.pickle

python -m profiling.memory_snapshot --model-size xl --batch-size 1 --context-length 2048 --mode train_step --dtype fp32 --warmup 5 --steps 1 --output results/a2p/memory/xl_2048_train.json --snapshot results/a2p/memory/xl_2048_train.pickle
```

如果 XL/2048 OOM：保存终端错误和失败配置，然后依题面按 XL/1024、Large/2048 顺序诊断。不得把 fallback 标成 XL/2048。

## 7. 整理 A2-P 结果

需要人工整理成：

```text
students/李畅松/assignments/A2-P/results/benchmark.csv
students/李畅松/assignments/A2-P/results/profile/trace_summary.csv
students/李畅松/assignments/A2-P/results/profile/run_metadata.json
students/李畅松/assignments/A2-P/results/mixed_precision.json
students/李畅松/assignments/A2-P/results/memory/peaks.csv
students/李畅松/assignments/A2-P/results/memory/run_metadata.json
```

同步代码：

```bash
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/SummerQuest-2026
python3 scripts/sync_a2p_submission.py --name '李畅松'
```

# A2-K：逐步运行

> 第 11–14 节所需的 GPU 驱动现已在上游 `assignment2-systems/student_scripts/a2k/` 中实现：
> `run_checkpointing.py`、`run_attention_benchmark.py`、`run_correctness.py` 和
> `run_flash_benchmark.py`。PyTorch tiled reference 与学生 Triton autograd 位于
> `assignment2-systems/cs336_systems/a2k/attention.py`。
>
> 当前非 GPU worker 只能验证导入、CLI、代码风格和 PyTorch reference；Triton kernel 会在
> 第一次 CUDA 调用时即时编译。正式性能实验前必须先在 RTX 4090 上完成第 10 节官方测试和
> 扩展正确性矩阵。正确性未通过时不得运行或提交性能数据。

## 8. 重要前提

当前已经完成纯 PyTorch tiled reference、学生 Triton forward/backward、activation checkpointing
和 attention benchmark 驱动。CPU 官方 reference forward/backward 测试通过；学生 Triton 的
CUDA 编译、数值正确性和性能仍必须在 RTX 4090 上实测，不能把 CPU reference 或 PyTorch
fallback 冒充 Triton 结果。

## 8.1 Reference 从提交目录同步到上游的方法

提交目录中的 reference 源文件位于：

```text
SummerQuest-2026/students/李畅松/assignments/A2-K/submission/
└── cs336_systems/a2k/
    ├── __init__.py
    └── attention.py
```

上游开发目标目录为：

```text
assignment2-systems/cs336_systems/a2k/
```

仅在上游还没有 A2-K 实现、需要从提交目录恢复 reference 时执行：

```bash
BASE=/inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012
cd "$BASE/assignment2-systems"
mkdir -p cs336_systems/a2k

cp \
  "$BASE/SummerQuest-2026/students/李畅松/assignments/A2-K/submission/cs336_systems/a2k/__init__.py" \
  cs336_systems/a2k/__init__.py

cp \
  "$BASE/SummerQuest-2026/students/李畅松/assignments/A2-K/submission/cs336_systems/a2k/attention.py" \
  cs336_systems/a2k/attention.py
```

不要整目录复制 `submission/cs336_systems/`，否则可能覆盖上游已经完成的 DDP、FSDP 和
sharded optimizer。不要复制 `__pycache__/` 或 `.pyc` 文件。

**当前上游的 `cs336_systems/a2k/attention.py` 已包含学生 Triton 实现，因此现在不要再次用
提交目录中的旧 reference 覆盖它。**上面的命令只用于解释初始同步或灾难恢复。

## 8.2 让官方测试指向 A2-K 包

不要用提交目录中的 `tests/adapters.py` 整份覆盖上游文件，因为上游 adapter 还包含 DDP、FSDP
和 sharded optimizer。只修改以下两个函数：

```python
def get_flashattention_autograd_function_pytorch() -> type:
    from cs336_systems.a2k.attention import FlashAttentionPytorch
    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> type:
    from cs336_systems.a2k.attention import FlashAttentionTriton
    return FlashAttentionTriton
```

当前上游 `tests/adapters.py` 已完成上述指向，无需重复修改。

## 9. 4090 环境和 allocator 检查

在每个正式脚本的第一处 CUDA tensor/model 创建之前放入：

```python
import torch
total_bytes = torch.cuda.get_device_properties(0).total_memory
limit_bytes = 23 * 1024**3
fraction = min(1.0, limit_bytes / total_bytes)
torch.cuda.set_per_process_memory_fraction(fraction, device=0)
print({"gpu": torch.cuda.get_device_name(0), "allocator_fraction": fraction})
```

运行环境脚本：

```bash
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/assignment2-systems
python student_scripts/a2k/check_environment.py
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate --format=csv,noheader
```

## 10. 官方正确性测试

```bash
mkdir -p local_results/a2k
python -m pytest -q tests/test_attention.py | tee local_results/a2k_unit_tests.txt
```

预期最终需要 PyTorch 和 Triton forward/backward 全部通过，即当前测试文件应报告 6 passed。
`skipped` 不能写成 `passed`。如果出现 Triton 编译错误、数值不一致、CUDA illegal access 或
其他失败，不要进入正式性能测试。

扩展正确性至少覆盖：

- causal 和 non-causal；
- head dimension 64、128；
- 多个 sequence length；
- BF16；
- 关闭 TF32 的 FP32；
- 输出 `O/L` 和梯度 `dQ/dK/dV` 的最大绝对/相对误差。

使用已经实现的扩展正确性驱动：

```bash
python student_scripts/a2k/run_correctness.py \
  --sequence-lengths 128 512 2048 \
  --output local_results/a2k_correctness.json
```

检查 JSON 的 `status` 必须为 `ok`，并检查每个 case 的 `output`、`dq`、`dk` 和 `dv` 的
`max_abs`/`max_rel`。只有官方测试和扩展矩阵都通过，才进入第 11–14 节。

## 11. Activation Checkpointing 正式矩阵

逐条运行，每条命令一个新进程：

```bash
mkdir -p local_results/a2k
for block in 0 1 2 4 8; do
  python student_scripts/a2k/run_checkpointing.py --model-size medium --num-layers 24 --batch-size 1 --context-length 1024 --dtype bf16 --checkpoint-block-size "$block" --warmup 3 --steps 5 --output "local_results/a2k/checkpoint_1024_b${block}.json"
done
```

`block=0` 表示不使用 checkpoint。找出 peak allocated 最低的成功配置，设为 `BEST`，然后运行：

```bash
BEST=1  # 改成实测最低显存的 block size
python student_scripts/a2k/run_checkpointing.py --model-size medium --num-layers 24 --batch-size 1 --context-length 2048 --dtype bf16 --checkpoint-block-size 0 --warmup 3 --steps 5 --output local_results/a2k/checkpoint_2048_b0.json
python student_scripts/a2k/run_checkpointing.py --model-size medium --num-layers 24 --batch-size 1 --context-length 2048 --dtype bf16 --checkpoint-block-size "$BEST" --warmup 3 --steps 5 --output "local_results/a2k/checkpoint_2048_b${BEST}.json"
```

OOM 也必须生成一条失败记录，包含配置、阶段、异常类型和峰值。

脚本会把成功或失败写入指定 JSON。成功记录包含 p20/p50/p80、peak allocated/reserved 和
最后一次 loss；OOM 记录的 `status` 为 `oom`，不得删除或改写成成功。

## 12. 显式 Attention 基线

运行显式 attention 基线：

```bash
for seq in 512 2048 8192; do
  for dim in 64 128; do
    python student_scripts/a2k/run_attention_benchmark.py --implementation eager --batch-size 1 --sequence-length "$seq" --head-dim "$dim" --dtype bf16 --causal --warmup-ms 100 --rep-ms 300 --output "local_results/a2k/attention_eager_s${seq}_d${dim}.json"
  done
done
```

必须记录 forward、backward、forward-backward 的 p20/p50/p80、peak allocated/reserved 和 OOM 状态。

这里的 `eager` 是显式的 `QK^T -> causal mask -> softmax -> PV`，会实际构造二次方大小的
score/probability 张量；它不是可能自动选择 fused kernel 的 PyTorch SDPA。8192 或 16384 OOM
时保留失败 JSON，不要缩小 shape 后冒充原配置。

## 13. torch.compile 对照

**环境兼容性：**当前配置若为 Python 3.13 + PyTorch 2.5.x，`torch.compile` 会直接报
`Dynamo is not supported on Python 3.13+`。这不是 attention benchmark 的 OOM 或代码错误。
正式采集 compile 对照时应使用 Python 3.12 环境（推荐保持 CUDA、驱动和 GPU 不变），或升级到
明确支持 Python 3.13 的 PyTorch 版本并先做一个小 shape smoke test。若无法更换环境，保留脚本
生成的 `status: unsupported` JSON，并在报告中说明 compile 对照因环境限制未采集；不要把它
标成 `status: ok`，也不要把 eager 结果冒充 compiled 结果。

本项目已建立专用于第 13–14 节的新环境：

```text
/inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/envs/summerquest-a2-py312
```

其已安装 Python 3.12.13、PyTorch 2.5.1+cu124、Triton 3.1.0 和两个 editable 项目包。
该组合与 GPU 节点现有 CUDA 12.4 驱动兼容，同时避免 Python 3.13 下 TorchDynamo 不可用的问题。
运行第 13–14 节前切换到该环境；其他已经采集完成的实验结果不需要重跑：

```bash
source /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/etc/profile.d/conda.sh
conda activate /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/OmniNextBench/miniconda3/envs/summerquest-a2-py312
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/assignment2-systems

python -c "import sys, torch, triton; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(triton.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
```

必须确认 Python 为 3.12、`cuda_available` 为 `True` 且 GPU 为 RTX 4090，再采集正式结果。

```bash
for spec in 512:64 2048:128 8192:128; do
  seq=${spec%:*}; dim=${spec#*:}
  python student_scripts/a2k/run_attention_benchmark.py --implementation compiled --batch-size 1 --sequence-length "$seq" --head-dim "$dim" --dtype bf16 --causal --warmup-ms 100 --rep-ms 300 --output "local_results/a2k/attention_compiled_s${seq}_d${dim}.json"
done
```

脚本必须把首次 compile/cold-start 时间和 steady-state 时间分开保存。

输出中的 `compile_cold_start_ms` 是第一次编译/执行时间，`metrics.*.p20_ms/p50_ms/p80_ms`
是预热后的 steady-state 测量。

## 14. FlashAttention 性能矩阵

在第 10 节全部通过后运行核心矩阵：

```bash
for seq in 512 2048 8192; do
  for dim in 64 128; do
    for causal in false true; do
      python student_scripts/a2k/run_flash_benchmark.py --batch-size 1 --sequence-length "$seq" --head-dim "$dim" --dtype bf16 --causal "$causal" --warmup-ms 100 --rep-ms 300 --output "local_results/a2k/flash_s${seq}_d${dim}_c${causal}.json"
    done
  done
done
```

再运行题面要求的 sequence 16384 边界配置。每个结果至少比较 eager、compiled 和学生 Triton；边界配置至少比较 eager 和学生 Triton。记录 forward/backward/forward-backward 的 p20/p50/p80、显存和 speedup。

每个 JSON 会保存 eager、compiled 和学生 Triton 的 cold start、steady-state 百分位、
peak allocated/reserved 以及相对 eager 的 speedup。sequence 16384 的显式 eager 很可能因
二次方 attention matrix OOM；这属于有效边界证据，必须保留 OOM 配置和异常阶段。

## 15. 整理 A2-K 结果

将真实结果汇总到：

```text
students/李畅松/assignments/A2-K/results/correctness.json
students/李畅松/assignments/A2-K/results/unit_tests.txt
students/李畅松/assignments/A2-K/results/checkpointing.csv
students/李畅松/assignments/A2-K/results/attention_baseline.csv
students/李畅松/assignments/A2-K/results/compile_comparison.csv
students/李畅松/assignments/A2-K/results/flash_benchmark.csv
students/李畅松/assignments/A2-K/results/memory_evidence.json
students/李畅松/assignments/A2-K/results/run_metadata.json
```

同步允许提交的代码：

```bash
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/SummerQuest-2026
python3 scripts/sync_a2k_submission.py --name '李畅松'
```

## 16. 最终检查

1. A2-P 放至少 3 张裁剪、脱敏、压缩并被 README 引用的图片。
2. A2-K 放至少 2 张图片。
3. 删除/替换所有 `GPU_NOT_RUN`。
4. 不提交 trace、snapshot、编译缓存、模型、数据或内部路径。
5. 填完两个 README 和飞书链接。

```bash
cd /inspire/hdd/project/exploration-topic/lichangsong-CZXS25250012/SummerQuest-2026
rg -n 'GPU_NOT_RUN|<填写>|<粘贴' students/李畅松/assignments/A2-P students/李畅松/assignments/A2-K
python3 scripts/validate_repo.py
git status --short
```
