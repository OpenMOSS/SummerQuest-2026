# CPU Smoke Training and Evaluation

- Date: 2026-07-14
- Working directory: `../assignment1-basics`
- Device: CPU
- Run name: `codex_smoke_cpu`

## Training command

```bash
PROFILE=tiny RUN_NAME=codex_smoke_cpu FORCE=1 \
  TRAIN_TEXT=tests/fixtures/tinystories_sample.txt \
  VALID_TEXT=tests/fixtures/tinystories_sample.txt \
  VOCAB_SIZE=300 CONTEXT_LENGTH=16 D_MODEL=32 NUM_LAYERS=1 \
  NUM_HEADS=4 D_FF=64 BATCH_SIZE=2 STEPS=2 \
  LOG_EVERY=1 EVAL_EVERY=1 EVAL_BATCHES=1 NUM_THREADS=2 \
  scripts/run_cpu_training.sh
```

## Training result

- Step 1: train loss `5.8035`, validation loss `5.8280`
- Step 2: train loss `5.7483`, validation loss `5.8011`
- Checkpoint: `runs/checkpoints/codex_smoke_cpu.pt`
- Train log: `runs/logs/codex_smoke_cpu_train.jsonl`

## Evaluation command

```bash
PROFILE=tiny RUN_NAME=codex_smoke_cpu NUM_THREADS=2 MAX_NEW_TOKENS=10 scripts/run_cpu_eval.sh
```

## Evaluation result

- Validation loss: `5.8017`
- Iteration: `2`

The generated sample is not meaningful because this smoke run only trains for two steps on a tiny fixture. It is used to verify that the CPU training and evaluation pipeline runs end to end.
