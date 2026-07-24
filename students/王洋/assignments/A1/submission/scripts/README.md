# Reproducible experiment scripts

Run every command from the repository root with `uv run`. Paths in JSON
configurations are interpreted relative to the current working directory.

## Tokenizer and data

Train the TinyStories tokenizer (add `--profile artifacts/tinystories/tokenizer.prof`
when profiling is needed):

```bash
uv run scripts/train_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output-dir artifacts/tokenizers/tinystories-10k \
  --vocab-size 10000 \
  --special-token '<|endoftext|>'
```

The artifact contains `tokenizer.json`, `vocab.json`, and `merges.txt`. Encode
train and validation data into mmap-compatible NumPy arrays:

```bash
uv run scripts/encode_dataset.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer artifacts/tokenizers/tinystories-10k \
  --output data/tokenized/tinystories_train.npy \
  --workers 8

uv run scripts/encode_dataset.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --tokenizer artifacts/tokenizers/tinystories-10k \
  --output data/tokenized/tinystories_valid.npy
```

`--dtype auto` uses `uint16` when the maximum vocabulary ID is at most 65,535
and otherwise uses `uint32`. Encoding uses a bounded text/token buffer and a
temporary raw token file per worker, so corpus size does not determine peak RAM
use. With `--workers N`, byte ranges are split only at occurrences of the
registered `--split-special-token` (default `<|endoftext|>`), and every process
loads its own tokenizer. If there are too few safe boundaries, the effective
worker count reported in the JSON sidecar is lower than requested. Parallel
encoding requires strict UTF-8 input; use `--workers 1` for the existing
`--errors replace` or `--errors ignore` behavior. The ordered worker parts are
merged atomically and produce the same token array as the single-process path.

Measure compression and end-to-end streaming encode throughput:

```bash
uv run scripts/sample_documents.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output artifacts/samples/tinystories_10docs.txt \
  --count 10
```

Then benchmark the fixed ten-document sample:

```bash
uv run scripts/benchmark_tokenizer.py \
  --tokenizer artifacts/tokenizers/tinystories-10k \
  --input artifacts/samples/tinystories_10docs.txt \
  --repeat 3 \
  --output artifacts/tinystories_tokenizer_benchmark.json
```

Run this command for each tokenizer/corpus pairing needed in the comparison.
Use the same `--max-chars` value for matched-prefix benchmarks.

## Model training and generation

Reproduce the ten-step toy SGD learning-rate comparison used in the written
answer:

```bash
uv run scripts/toy_sgd.py --output artifacts/toy_sgd.json
```

Start a configured experiment:

```bash
uv run scripts/train_lm.py --config configs/tinystories_baseline.json
```

For learning-rate and effective-batch sweeps, use `--learning-rate`,
`--min-learning-rate`, `--warmup-iters`, `--cosine-cycle-iters`,
`--micro-batch-size`, and `--gradient-accumulation-steps`. Evaluation, logging,
and checkpoint intervals also have CLI overrides; every override is captured in
`config.resolved.json` and the checkpoint.

Resume its latest checkpoint, optionally increasing the step target:

```bash
uv run scripts/train_lm.py \
  --config configs/tinystories_baseline.json \
  --resume \
  --max-steps 12000
```

The output directory contains `config.resolved.json`, `metrics.jsonl`, and
`latest.pt`. Each input `.npy` must retain the `.npy.json` sidecar written by
`encode_dataset.py`; training checks its shape, dtype, token count, vocabulary
size, and tokenizer hashes, requires train/validation tokenizer identity, and
stores that provenance in the resolved config and every checkpoint.

Checkpoints contain model/optimizer state, the completed gradient step,
processed-token count, cumulative wall time, resolved configuration, data
provenance, and Python/NumPy/Torch RNG states. Resume permits a new output/device,
a larger `max_steps`, and different logging/evaluation/checkpoint intervals, but
rejects changes to model, data, tokenizer identity, optimizer, schedule,
precision, seed, effective training batch, or gradient clipping. Before append,
an existing metrics tail beyond the checkpoint is atomically removed, so step
and processed-token fields stay monotonic. Changing `max_steps` does not implicitly
change an explicit cosine-cycle endpoint.

JSONL events include `run_start`, `train`, `validation`, `checkpoint`, and
`run_end` (or `diverged`). They are strict JSON: a non-finite diagnostic is stored
as `null` with its original `NaN`/`+Infinity`/`-Infinity` spelling under
`nonfinite_fields`. A non-finite loss or global gradient norm stops before
`optimizer.step()` and writes `diverged.pt` with the still-unupdated model and
optimizer plus diagnostic metadata.

Summarize one or more logs for report tables:

```bash
uv run scripts/summarize_metrics.py \
  runs/tinystories_baseline/metrics.jsonl \
  --output artifacts/tinystories_baseline_summary.json
```

Plot one run, or overlay several runs by listing multiple JSONL files. The SVG
renderer uses only the Python standard library:

```bash
uv run scripts/plot_metrics.py \
  runs/tinystories_baseline/metrics.jsonl \
  runs/tinystories_nope/metrics.jsonl \
  --label baseline \
  --label NoPE \
  --x-axis processed_tokens \
  --output assets/baseline_vs_nope.svg
```

Generate with temperature and nucleus sampling:

```bash
uv run scripts/generate.py \
  --tokenizer artifacts/tokenizers/tinystories-10k \
  --config runs/tinystories_baseline/config.resolved.json \
  --checkpoint runs/tinystories_baseline/latest.pt \
  --prompt 'Once upon a time' \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 1337 \
  --output runs/tinystories_baseline/generation.txt \
  --metadata-output runs/tinystories_baseline/generation.json
```

Generation verifies that the tokenizer vocabulary/merge SHA-256 exactly matches
the provenance bound to the resolved config or checkpoint. Legacy unbound runs
are rejected by default; use `--allow-unbound-tokenizer` only after manually
auditing the tokenizer identity.

The supplied configurations are:

- `tinystories_baseline.json`
- `tinystories_low_resource.json` (40.96M-token baseline architecture)
- `cpu_debug.json`
- `smoke_test.json` (two CPU steps; override its data and output paths for CI)
- `ablation_baseline.json` (the exact 1,000-step control used by the ablations)
- `ablation_remove_rmsnorm.json` (LR `3e-3`; expected to diverge early)
- `ablation_remove_rmsnorm_low_lr.json` (stable LR `1e-4` control)
- `ablation_post_norm.json`
- `ablation_nope.json`
- `ablation_silu_matched.json` (SiLU width 2,016 matches the baseline SwiGLU
  parameter count)
- `owt_baseline.json`
