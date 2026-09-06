#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
EVALUATOR="${EVALUATOR:-${ROOT}/lightx2v_train/tools/eval_fastwam_libero_checkpoint.py}"
WEIGHT="${WEIGHT:-${ROOT}/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_16gpu_mbs48_nogc/exports/checkpoint-000030000-student.pt}"
POLICY_CONFIG="${POLICY_CONFIG:-${ROOT}/configs/fastwam/libero_i2va_dmd_1step.json}"
DATASET_STATS="${DATASET_STATS:-/mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224_dataset_stats.json}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs_1/charles/models/Wan2.2-TI2V-5B}"
LIBERO_ROOT="${LIBERO_ROOT:-${ROOT}/lightx2v_ros/src/simulator/simulator/libero_node/LIBERO}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
TASKS_PER_SHARD="${TASKS_PER_SHARD:-5}"
WORKERS_PER_DEVICE="${WORKERS_PER_DEVICE:-1}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-libero_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/evaluation_outputs/libero/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/driver.log}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$2" ]] || fail "$1 file not found: $2"
}

require_dir() {
    [[ -d "$2" ]] || fail "$1 directory not found: $2"
}

require_positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer: $2"
}

[[ -x "$PYTHON" ]] || fail "Python executable not found or not executable: $PYTHON"
require_dir "LightX2V project" "$ROOT"
require_file "evaluation driver" "$EVALUATOR"
require_file "weight" "$WEIGHT"
require_file "policy config" "$POLICY_CONFIG"
require_file "dataset stats" "$DATASET_STATS"
require_dir "base model" "$MODEL_PATH"
require_dir "LIBERO project" "$LIBERO_ROOT"
require_positive_int "EPISODES_PER_TASK" "$EPISODES_PER_TASK"
require_positive_int "TASKS_PER_SHARD" "$TASKS_PER_SHARD"
require_positive_int "WORKERS_PER_DEVICE" "$WORKERS_PER_DEVICE"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer: $SEED"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1: $DRY_RUN"
read -r -a DEVICES <<< "$GPU_IDS"
[[ "${#DEVICES[@]}" -gt 0 ]] || fail "GPU_IDS must contain at least one device"
for device in "${DEVICES[@]}"; do
    [[ "$device" =~ ^[0-9]+$ ]] || fail "GPU_IDS must contain non-negative integers: $GPU_IDS"
done

mkdir -p -- "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/lightx2v_train:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "$PYTHON" "$EVALUATOR"
    --adapter "$WEIGHT"
    --output-root "$OUTPUT_DIR"
    --model-path "$MODEL_PATH"
    --policy-config "$POLICY_CONFIG"
    --dataset-stats "$DATASET_STATS"
    --libero-root "$LIBERO_ROOT"
    --benchmarks libero_spatial libero_object libero_goal libero_10
    --devices "${DEVICES[@]}"
    --workers-per-device "$WORKERS_PER_DEVICE"
    --episodes-per-task "$EPISODES_PER_TASK"
    --tasks-per-shard "$TASKS_PER_SHARD"
    --seed "$SEED"
    --expected-action-infer-steps 1
)
COMMAND+=("$@")

printf 'Mode: %s\n' "$([[ "$DRY_RUN" == "1" ]] && printf DRY_RUN || printf EVALUATION)"
printf 'Repository root: %s\n' "$ROOT"
printf 'Output directory: %s\n' "$OUTPUT_DIR"
printf 'Environment:\n'
printf '  PYTHON=%s\n' "$PYTHON"
printf '  PYTHONPATH=%s\n' "$PYTHONPATH"
printf '  GPU_IDS=%s\n' "$GPU_IDS"
printf 'Command:\n  '
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

cd -- "$ROOT"
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
