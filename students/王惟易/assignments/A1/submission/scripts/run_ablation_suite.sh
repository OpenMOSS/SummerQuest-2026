#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUITE_DIR="runs/ablation_suite"
STATUS_FILE="$SUITE_DIR/status.tsv"
ARCHIVE_PATH="$SUITE_DIR/a1-results.tar.gz"
EXPECTED_STEP=2000

mkdir -p "$SUITE_DIR"

exec 9>"$SUITE_DIR/runner.lock"
if ! flock -n 9; then
    printf '%s\tSUITE\talready-running\n' "$(date -Is)" >> "$STATUS_FILE"
    exit 0
fi

rm -f "$SUITE_DIR/COMPLETE" "$SUITE_DIR/FAILED" "$SUITE_DIR/RUNNING"
touch "$SUITE_DIR/RUNNING"

CONFIGS=(
    "configs/tinystories_ablation_control.json"
    "configs/tinystories_ablation_no_norm.json"
    "configs/tinystories_ablation_no_norm_low_lr.json"
    "configs/tinystories_ablation_post_norm.json"
    "configs/tinystories_ablation_nope.json"
    "configs/tinystories_ablation_silu.json"
)

RUN_NAMES=(
    "tinystories_ablation_control"
    "tinystories_ablation_no_norm"
    "tinystories_ablation_no_norm_low_lr"
    "tinystories_ablation_post_norm"
    "tinystories_ablation_nope"
    "tinystories_ablation_silu"
)

printf '%s\tSUITE\tstarted\n' "$(date -Is)" >> "$STATUS_FILE"
suite_failed=0

for index in "${!CONFIGS[@]}"; do
    config_path="${CONFIGS[$index]}"
    run_name="${RUN_NAMES[$index]}"
    metrics_path="runs/$run_name/metrics.jsonl"

    if [[ -f "$metrics_path" ]] && tail -n 1 "$metrics_path" | grep -q "\"step\": $EXPECTED_STEP"; then
        printf '%s\t%s\tskipped-complete\n' "$(date -Is)" "$run_name" >> "$STATUS_FILE"
        continue
    fi

    printf '%s\t%s\tstarted\n' "$(date -Is)" "$run_name" >> "$STATUS_FILE"
    if uv run python scripts/train.py --config "$config_path"; then
        printf '%s\t%s\tsucceeded\n' "$(date -Is)" "$run_name" >> "$STATUS_FILE"
    else
        exit_code=$?
        printf '%s\t%s\tfailed:%s\n' "$(date -Is)" "$run_name" "$exit_code" >> "$STATUS_FILE"
        suite_failed=1
    fi
done

archive_inputs=("${CONFIGS[@]}" "$STATUS_FILE")
while IFS= read -r -d '' path; do
    archive_inputs+=("$path")
done < <(find runs -mindepth 2 -maxdepth 2 -type f -name metrics.jsonl -print0)

if [[ -d "$SUITE_DIR/historical_configs" ]]; then
    while IFS= read -r -d '' path; do
        archive_inputs+=("$path")
    done < <(find "$SUITE_DIR/historical_configs" -maxdepth 1 -type f -name '*.json' -print0)
fi

if tar -czf "$ARCHIVE_PATH" "${archive_inputs[@]}"; then
    printf '%s\tSUITE\tarchived\n' "$(date -Is)" >> "$STATUS_FILE"
    if (( suite_failed == 0 )); then
        touch "$SUITE_DIR/COMPLETE"
    else
        touch "$SUITE_DIR/FAILED"
    fi
    rm -f "$SUITE_DIR/RUNNING"
else
    printf '%s\tSUITE\tarchive-failed\n' "$(date -Is)" >> "$STATUS_FILE"
    rm -f "$SUITE_DIR/RUNNING"
    touch "$SUITE_DIR/FAILED"
    exit 1
fi
