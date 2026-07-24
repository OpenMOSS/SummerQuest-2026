# GPU Experiments

All runs used the implementation in `../assignment1-basics` and wrote raw artifacts under `runs/`. Checkpoints, encoded corpora and tokenizer files are not committed. Normalized JSONL logs are committed next to this summary:

- `train_tinystories.jsonl`
- `train_owt.jsonl`
- `ablation_no_rmsnorm.jsonl`, `ablation_post_norm.jsonl`, `ablation_no_rope.jsonl`, `ablation_silu_ffn.jsonl`
- `lr_sweep/*.jsonl`
- `batch_size/*.jsonl`
- `summary.json`

Each step record includes `step`, `wall_clock_sec`, `train_loss` and `lr`; validation records also include `valid_loss`.

## TinyStories Full-Model Runs

Common configuration unless noted otherwise:

```text
vocab_size=10000
context_length=256
d_model=512
d_ff=1344
num_layers=4
num_heads=16
rope_theta=10000
batch_size=128
steps=10000
lr=3e-4
min_lr=3e-5
warmup_steps=100
grad_clip=1.0
device=cuda
tf32=1
```

| run | steps | train loss | valid loss | elapsed seconds | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `tinystories_gpu_baseline` | 10,000 | 1.5004 | 1.5109 | 1014.2 | baseline |
| `tinystories_gpu_target45_continue20k` | 20,000 | 1.5047 | 1.4552 | 1024.0 | continued baseline longer; near 1.45 |
| `tinystories_gpu_target45_highlr10k` | 10,000 | 1.3377 | 1.3517 | 817.8 | resumed from `lr_3em3` 2000-step checkpoint; meets 1.45 |

## Architecture Ablations

All ablations changed one architectural variable relative to the baseline and kept the same data, seed, optimizer settings and 10,000-step budget.

| run | change | train loss | valid loss | elapsed seconds |
| --- | --- | ---: | ---: | ---: |
| `tinystories_gpu_baseline` | baseline | 1.5004 | 1.5109 | 1014.2 |
| `tinystories_gpu_no_rmsnorm` | remove RMSNorm | 1.5295 | 1.5400 | 926.7 |
| `tinystories_gpu_post_norm` | use Post-Norm | 1.5039 | 1.5114 | 1011.4 |
| `tinystories_gpu_no_rope` | disable RoPE | 1.6011 | 1.6039 | 921.6 |
| `tinystories_gpu_silu_ffn` | SiLU FFN with `silu_d_ff=2016` | 1.5429 | 1.5481 | 995.7 |

RoPE had the largest measured effect in this sweep. Removing RMSNorm and replacing SwiGLU with a parameter-matched SiLU FFN also hurt validation loss; Post-Norm was close to baseline over this short horizon.

## Learning-Rate Sweep

All LR sweep runs used TinyStories, batch 128, 2000 steps, same architecture and seed.

| run | max LR | min LR | train loss | valid loss | elapsed seconds | observation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tinystories_gpu_lr_1em4` | 1e-4 | 1e-5 | 2.4746 | 2.4335 | 211.9 | under-trained |
| `tinystories_gpu_lr_3em4` | 3e-4 | 3e-5 | 1.9662 | 1.9387 | 202.9 | stable |
| `tinystories_gpu_lr_1em3` | 1e-3 | 1e-4 | 1.6514 | 1.6411 | 203.0 | stable, faster |
| `tinystories_gpu_lr_3em3` | 3e-3 | 3e-4 | 1.5626 | 1.5629 | 202.9 | best in 2k sweep |
| `tinystories_gpu_lr_1em1` | 1e-1 | 1e-2 | 3.8597 | 3.8268 | 204.5 | unstable/divergent |

## Batch-Size Sweep

Batch-size sweep used TinyStories and the baseline architecture. Batch 1-32 were short 1000-step probes; batch 64 and 128 used 10000 steps, so final losses should be interpreted with the processed-token column.

| batch | steps | processed tokens | train loss | valid loss | elapsed seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 256,000 | 4.3602 | 3.6470 | 16.3 |
| 2 | 1,000 | 512,000 | 3.6064 | 3.2888 | 16.6 |
| 4 | 1,000 | 1,024,000 | 2.9885 | 3.0544 | 16.9 |
| 8 | 1,000 | 2,048,000 | 2.7226 | 2.8968 | 17.3 |
| 16 | 1,000 | 4,096,000 | 2.6724 | 2.6904 | 19.0 |
| 32 | 1,000 | 8,192,000 | 2.6280 | 2.5129 | 30.3 |
| 64 | 10,000 | 163,840,000 | 1.5438 | 1.5675 | 542.0 |
| 128 | 10,000 | 327,680,000 | 1.5004 | 1.5109 | 1011.4 |

## OWT Full-Model Run

OWT used the same architecture as TinyStories with `vocab_size=32000`, `batch_size=32`, and 10000 steps.

| run | train tokens | valid tokens | processed tokens | train loss | valid loss | elapsed seconds | tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `owt_gpu_full` | 262,963,181 | 66,401,098 | 81,920,000 | 4.6059 | 4.6234 | 395.2 | 207k |

OWT per-token loss is not directly comparable to TinyStories because the corpus and tokenizer differ.
