#!/usr/bin/env bash
# Run the OWT tokenizer follow-up stages after tokenizer training finishes.
#
# This script is deliberately resumable: an existing stage output is reused only
# after it passes the same validation applied to a newly generated output.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOKENIZER="artifacts/owt_tokenizer.json"
TOKENIZER_SUMMARY="artifacts/owt_tokenizer_summary.json"
TINY_TOKENIZER="artifacts/tinystories_tokenizer.json"
TINY_THROUGHPUT_METRICS="artifacts/tinystories_train_tokenizer_metrics.json"
OWT_TRAIN="data/owt_train.txt"
OWT_VALID="data/owt_valid.txt"
TINY_TRAIN="data/TinyStoriesV2-GPT4-train.txt"

OWT_VALID_METRICS="artifacts/owt_validation_tokenizer_metrics.json"
TINY_COMPARISON="artifacts/tinystories_tokenizer_comparison.json"
OWT_COMPARISON="artifacts/owt_tokenizer_comparison.json"
OWT_VALID_ARRAY="data/owt_validation.npy"
OWT_TRAIN_ARRAY="data/owt_train.npy"
OWT_VALID_ENCODING="artifacts/owt_validation_encoding.json"
OWT_TRAIN_ENCODING="artifacts/owt_train_encoding.json"

LOG_DIR="artifacts/console/owt_postprocess"
STATE_DIR="artifacts/owt_postprocess_state"
COMPLETE_MARKER="artifacts/owt_encoding_complete"
COMPLETE_REPORT="artifacts/owt_postprocess_complete.json"
FAILED_MARKER="artifacts/owt_postprocess_failed.json"
WAIT_SECONDS="${OWT_POSTPROCESS_WAIT_SECONDS:-30}"
WORKERS="${OWT_POSTPROCESS_WORKERS:-8}"
CACHE_SIZE="${OWT_POSTPROCESS_CACHE_SIZE:-32768}"
CURRENT_STAGE="startup"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$STATE_DIR"

write_failure_marker() {
  local exit_code=$?
  if [[ $DRY_RUN -eq 0 ]]; then
    uv run python - "$FAILED_MARKER" "$CURRENT_STAGE" "$exit_code" <<'PY' || true
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, stage, exit_code = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "status": "failed",
            "stage": stage,
            "exit_code": int(exit_code),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  fi
  echo "OWT post-processing failed in stage '$CURRENT_STAGE' (exit $exit_code)." >&2
  exit "$exit_code"
}
trap write_failure_marker ERR

run_logged() {
  local stage=$1
  shift
  CURRENT_STAGE=$stage
  echo "[$(date -Is)] START $stage"
  "$@" 2>&1 | tee "$LOG_DIR/$stage.log"
  echo "[$(date -Is)] DONE  $stage"
}

validate_tokenizer() {
  uv run python - "$TOKENIZER" "$TOKENIZER_SUMMARY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

tokenizer_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
tokenizer = Tokenizer.load(tokenizer_path)
ids = sorted(tokenizer.vocab)

assert summary["input_name"] == "owt_train.txt", summary
assert summary["vocab_size"] == 32_000, summary
assert summary["num_merges"] == 31_743, summary
assert summary["output_name"] == tokenizer_path.name, summary
assert summary["special_tokens"] == ["<|endoftext|>"], summary
assert len(tokenizer.vocab) == 32_000
assert len(tokenizer.merge_ranks) == 31_743
assert ids == list(range(32_000)), "token IDs are not contiguous in [0, 32000)"
special_encoding = tokenizer.encode("<|endoftext|>")
assert len(special_encoding) == 1, special_encoding
assert tokenizer.decode(special_encoding) == "<|endoftext|>"
print(json.dumps({"status": "valid", "vocab_size": len(ids), "num_merges": len(tokenizer.merge_ranks)}))
PY
}

validate_metrics() {
  local path=$1
  local expected_input=$2
  uv run python - "$path" "$expected_input" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert metrics["input_name"] == sys.argv[2], metrics
for key in ("bytes", "tokens", "bytes_per_token", "tokens_per_sec", "megabytes_per_sec", "elapsed_sec"):
    value = metrics[key]
    assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0, (key, value)
assert 0 <= metrics["longest_token_id"] < 32_000, metrics
print(json.dumps({"status": "valid", "path": sys.argv[1], "tokens": metrics["tokens"]}))
PY
}

validate_comparison() {
  local path=$1
  local expected_input=$2
  uv run python - "$path" "$expected_input" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["input_name"] == sys.argv[2], result
assert result["document_count"] == 10, result
assert set(result["tokenizers"]) == {"tinystories_10k", "owt_32k"}, result
for metrics in result["tokenizers"].values():
    assert len(metrics["documents"]) == 10, metrics
    assert metrics["total_bytes"] > 0 and metrics["total_tokens"] > 0, metrics
    assert math.isfinite(metrics["bytes_per_token"]) and metrics["bytes_per_token"] > 0, metrics
assert set(result["throughput_extrapolations"]) == {"tinystories_10k", "owt_32k"}, result
for extrapolation in result["throughput_extrapolations"].values():
    assert extrapolation["target_gb"] == 825.0, extrapolation
    assert extrapolation["estimated_seconds"] > 0, extrapolation
print(json.dumps({"status": "valid", "path": sys.argv[1], "documents": 10, "target_gb": 825.0}))
PY
}

validate_encoding() {
  local array_path=$1
  local summary_path=$2
  local expected_input=$3
  uv run python - "$array_path" "$summary_path" "$expected_input" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

array_path = Path(sys.argv[1])
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
tokens = np.load(array_path, mmap_mode="r")
assert summary["input_name"] == sys.argv[3], summary
assert summary["output_name"] == array_path.name, summary
assert summary["dtype"] == "uint16", summary
assert tokens.dtype == np.uint16, tokens.dtype
assert tokens.ndim == 1 and tokens.size > 0, tokens.shape
assert summary["tokens"] == tokens.size, (summary["tokens"], tokens.size)
minimum = int(tokens.min())
maximum = int(tokens.max())
assert 0 <= minimum <= maximum < 32_000, (minimum, maximum)
assert array_path.stat().st_size >= tokens.size * tokens.dtype.itemsize
print(json.dumps({"status": "valid", "path": str(array_path), "tokens": tokens.size, "min": minimum, "max": maximum}))
PY
}

if [[ $DRY_RUN -eq 1 ]]; then
  cat <<EOF
OWT post-processing dry run (no long command executed)
  wait for: $TOKENIZER and $TOKENIZER_SUMMARY
  1. validate OWT tokenizer (32,000 vocab; 31,743 merges; contiguous uint16-compatible IDs)
  2. benchmark OWT tokenizer on $OWT_VALID -> $OWT_VALID_METRICS
  3. compare TinyStories/OWT tokenizers on 10 TinyStories documents -> $TINY_COMPARISON
  4. compare TinyStories/OWT tokenizers on 10 OWT documents -> $OWT_COMPARISON
  5. encode and validate OWT validation -> $OWT_VALID_ARRAY
  6. encode and validate OWT train -> $OWT_TRAIN_ARRAY
  7. write completion report -> $COMPLETE_REPORT; then touch gate -> $COMPLETE_MARKER
EOF
  exit 0
fi

rm -f "$COMPLETE_MARKER" "$COMPLETE_REPORT" "$FAILED_MARKER"

for required in "$TINY_TOKENIZER" "$TINY_THROUGHPUT_METRICS" "$OWT_TRAIN" "$OWT_VALID" "$TINY_TRAIN"; do
  if [[ ! -s "$required" ]]; then
    echo "Required input is missing or empty: $required" >&2
    false
  fi
done

CURRENT_STAGE="wait_for_owt_tokenizer"
while [[ ! -s "$TOKENIZER" || ! -s "$TOKENIZER_SUMMARY" ]]; do
  echo "[$(date -Is)] Waiting for $TOKENIZER and $TOKENIZER_SUMMARY ..."
  sleep "$WAIT_SECONDS"
done
validate_tokenizer >"$LOG_DIR/tokenizer_validation.log" 2>&1
cat "$LOG_DIR/tokenizer_validation.log"
touch "$STATE_DIR/tokenizer_validated.done"

if ! validate_metrics "$OWT_VALID_METRICS" "owt_valid.txt" >"$LOG_DIR/validate_owt_valid_benchmark.log" 2>&1; then
  run_logged "benchmark_owt_valid" uv run python scripts/benchmark_tokenizer.py \
    --tokenizer "$TOKENIZER" \
    --input "$OWT_VALID" \
    --output "$OWT_VALID_METRICS" \
    --workers "$WORKERS" \
    --cache-size "$CACHE_SIZE" \
    --progress
fi
validate_metrics "$OWT_VALID_METRICS" "owt_valid.txt" | tee "$LOG_DIR/validate_owt_valid_benchmark.log"
touch "$STATE_DIR/benchmark_owt_valid.done"

if ! validate_comparison "$TINY_COMPARISON" "TinyStoriesV2-GPT4-train.txt" >"$LOG_DIR/validate_tiny_comparison.log" 2>&1; then
  run_logged "compare_tinystories_documents" uv run python scripts/analyze_tokenizer_samples.py \
    --input "$TINY_TRAIN" \
    --tokenizer "tinystories_10k=$TINY_TOKENIZER" \
    --tokenizer "owt_32k=$TOKENIZER" \
    --documents 10 \
    --throughput-json "tinystories_10k=$TINY_THROUGHPUT_METRICS" \
    --throughput-json "owt_32k=$OWT_VALID_METRICS" \
    --target-gb 825 \
    --output "$TINY_COMPARISON"
fi
validate_comparison "$TINY_COMPARISON" "TinyStoriesV2-GPT4-train.txt" | tee "$LOG_DIR/validate_tiny_comparison.log"
touch "$STATE_DIR/compare_tinystories_documents.done"

if ! validate_comparison "$OWT_COMPARISON" "owt_train.txt" >"$LOG_DIR/validate_owt_comparison.log" 2>&1; then
  run_logged "compare_owt_documents" uv run python scripts/analyze_tokenizer_samples.py \
    --input "$OWT_TRAIN" \
    --tokenizer "tinystories_10k=$TINY_TOKENIZER" \
    --tokenizer "owt_32k=$TOKENIZER" \
    --documents 10 \
    --throughput-json "tinystories_10k=$TINY_THROUGHPUT_METRICS" \
    --throughput-json "owt_32k=$OWT_VALID_METRICS" \
    --target-gb 825 \
    --output "$OWT_COMPARISON"
fi
validate_comparison "$OWT_COMPARISON" "owt_train.txt" | tee "$LOG_DIR/validate_owt_comparison.log"
touch "$STATE_DIR/compare_owt_documents.done"

if ! validate_encoding "$OWT_VALID_ARRAY" "$OWT_VALID_ENCODING" "owt_valid.txt" >"$LOG_DIR/validate_owt_valid_encoding.log" 2>&1; then
  run_logged "encode_owt_valid" uv run python scripts/encode_dataset.py \
    --tokenizer "$TOKENIZER" \
    --input "$OWT_VALID" \
    --output "$OWT_VALID_ARRAY" \
    --summary "$OWT_VALID_ENCODING" \
    --cache-size "$CACHE_SIZE" \
    --progress
fi
validate_encoding "$OWT_VALID_ARRAY" "$OWT_VALID_ENCODING" "owt_valid.txt" | tee "$LOG_DIR/validate_owt_valid_encoding.log"
touch "$STATE_DIR/encode_owt_valid.done"

if ! validate_encoding "$OWT_TRAIN_ARRAY" "$OWT_TRAIN_ENCODING" "owt_train.txt" >"$LOG_DIR/validate_owt_train_encoding.log" 2>&1; then
  run_logged "encode_owt_train" uv run python scripts/encode_dataset.py \
    --tokenizer "$TOKENIZER" \
    --input "$OWT_TRAIN" \
    --output "$OWT_TRAIN_ARRAY" \
    --summary "$OWT_TRAIN_ENCODING" \
    --cache-size "$CACHE_SIZE" \
    --progress
fi
validate_encoding "$OWT_TRAIN_ARRAY" "$OWT_TRAIN_ENCODING" "owt_train.txt" | tee "$LOG_DIR/validate_owt_train_encoding.log"
touch "$STATE_DIR/encode_owt_train.done"

CURRENT_STAGE="write_complete_marker"
uv run python - "$COMPLETE_REPORT" "$OWT_VALID_METRICS" "$TINY_COMPARISON" "$OWT_COMPARISON" "$OWT_VALID_ENCODING" "$OWT_TRAIN_ENCODING" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_path = Path(sys.argv[1])
artifacts = [str(Path(path)) for path in sys.argv[2:]]
output_path.write_text(
    json.dumps(
        {
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "validated_artifacts": artifacts,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
touch "$COMPLETE_MARKER"
rm -f "$FAILED_MARKER"
echo "[$(date -Is)] OWT post-processing complete: $COMPLETE_MARKER (report: $COMPLETE_REPORT)"
