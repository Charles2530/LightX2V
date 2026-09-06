#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/afs_1/charles/env/miniconda3/envs/lightx2v_libero_plus/bin/python}"
LAUNCHER_ROOT="${LAUNCHER_ROOT:-/mnt/afs_1/charles/codes/LightX2V_fastwam_20step_f8164573}"
EVALUATION_ROOT="${EVALUATION_ROOT:-/mnt/afs_1/charles/codes/LightX2V_fastwam_eval_fcf10f6a}"
EVALUATOR="${EVALUATOR:-${LAUNCHER_ROOT}/lightx2v_train/tools/eval_fastwam_libero_shared_checkpoint.py}"
SHARED_POLICY_EVALUATOR="${SHARED_POLICY_EVALUATOR:-${EVALUATION_ROOT}/lightx2v_train/tools/eval_fastwam_libero_shared_policy.py}"
WEIGHT="${WEIGHT:-/mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt}"
POLICY_CONFIG="${POLICY_CONFIG:-${ROOT}/configs/fastwam/libero_i2va.json}"
DATASET_STATS="${DATASET_STATS:-/mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224_dataset_stats.json}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs_1/charles/models/Wan2.2-TI2V-5B}"
LIBERO_ROOT="${LIBERO_ROOT:-/mnt/afs_1/charles/codes/LIBERO-plus}"
NVIDIA_EGL_ROOT="${NVIDIA_EGL_ROOT:-/mnt/afs_1/charles/env/nvidia-egl-550.90.07/root}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
EGL_DEVICE_OVERRIDES="${EGL_DEVICE_OVERRIDES-5=4 7=6}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
TASKS_PER_SHARD="${TASKS_PER_SHARD:-1}"
ENV_WORKERS_PER_DEVICE="${ENV_WORKERS_PER_DEVICE:-12}"
ACTION_INFER_STEPS="${ACTION_INFER_STEPS:-20}"
ACTIONS_PER_PLAN="${ACTIONS_PER_PLAN:-10}"
PROMPT_CACHE_LIMIT="${PROMPT_CACHE_LIMIT:-256}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-libero_plus_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${LIBERO_ROOT}/eval_results/fastwam_20step_1trial_official/one_command/${RUN_ID}}"
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
require_dir "successful launcher project" "$LAUNCHER_ROOT"
require_dir "frozen evaluation project" "$EVALUATION_ROOT"
require_file "evaluation driver" "$EVALUATOR"
require_file "shared-policy evaluator" "$SHARED_POLICY_EVALUATOR"
require_file "weight" "$WEIGHT"
require_file "policy config" "$POLICY_CONFIG"
require_file "dataset stats" "$DATASET_STATS"
require_dir "base model" "$MODEL_PATH"
require_dir "LIBERO-Plus project" "$LIBERO_ROOT"
require_file "LIBERO-Plus official metric config" "${LIBERO_ROOT}/libero/configs/eval/default.yaml"
require_file "LIBERO-Plus official metric" "${LIBERO_ROOT}/libero/lifelong/metric.py"
require_dir "NVIDIA EGL runtime" "$NVIDIA_EGL_ROOT"
require_file "NVIDIA EGL vendor config" "${NVIDIA_EGL_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
require_positive_int "EPISODES_PER_TASK" "$EPISODES_PER_TASK"
require_positive_int "TASKS_PER_SHARD" "$TASKS_PER_SHARD"
require_positive_int "ENV_WORKERS_PER_DEVICE" "$ENV_WORKERS_PER_DEVICE"
require_positive_int "ACTION_INFER_STEPS" "$ACTION_INFER_STEPS"
require_positive_int "ACTIONS_PER_PLAN" "$ACTIONS_PER_PLAN"
require_positive_int "PROMPT_CACHE_LIMIT" "$PROMPT_CACHE_LIMIT"
require_positive_int "STARTUP_TIMEOUT" "$STARTUP_TIMEOUT"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer: $SEED"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1: $DRY_RUN"
read -r -a DEVICES <<< "$GPU_IDS"
[[ "${#DEVICES[@]}" -gt 0 ]] || fail "GPU_IDS must contain at least one device"
for device in "${DEVICES[@]}"; do
    [[ "$device" =~ ^[0-9]+$ ]] || fail "GPU_IDS must contain non-negative integers: $GPU_IDS"
done

mkdir -p -- "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export FASTWAM_EVALUATION_ROOT="$EVALUATION_ROOT"
export PYTHONPATH="${EVALUATION_ROOT}/lightx2v_train:${EVALUATION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "$PYTHON" "$EVALUATOR"
    --adapter "$WEIGHT"
    --output-root "$OUTPUT_DIR"
    --model-path "$MODEL_PATH"
    --policy-config "$POLICY_CONFIG"
    --dataset-stats "$DATASET_STATS"
    --libero-root "$LIBERO_ROOT"
    --devices "${DEVICES[@]}"
    --env-workers-per-device "$ENV_WORKERS_PER_DEVICE"
    --episodes-per-task "$EPISODES_PER_TASK"
    --tasks-per-shard "$TASKS_PER_SHARD"
    --seed "$SEED"
    --expected-action-infer-steps "$ACTION_INFER_STEPS"
    --expected-actions-per-plan "$ACTIONS_PER_PLAN"
    --prompt-cache-limit "$PROMPT_CACHE_LIMIT"
    --startup-timeout "$STARTUP_TIMEOUT"
    --nvidia-egl-root "$NVIDIA_EGL_ROOT"
)
for override in $EGL_DEVICE_OVERRIDES; do
    override_device="${override%%=*}"
    [[ "$override" == *=* && "$override_device" =~ ^[0-9]+$ ]] || fail "Invalid EGL_DEVICE_OVERRIDES entry: $override"
    for device in "${DEVICES[@]}"; do
        if [[ "$device" == "$override_device" ]]; then
            COMMAND+=(--egl-device-override "$override")
            break
        fi
    done
done
COMMAND+=("$@")

printf 'Mode: %s\n' "$([[ "$DRY_RUN" == "1" ]] && printf DRY_RUN || printf EVALUATION)"
printf 'Repository root: %s\n' "$ROOT"
printf 'Output directory: %s\n' "$OUTPUT_DIR"
printf 'Environment:\n'
printf '  PYTHON=%s\n' "$PYTHON"
printf '  FASTWAM_EVALUATION_ROOT=%s\n' "$FASTWAM_EVALUATION_ROOT"
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
