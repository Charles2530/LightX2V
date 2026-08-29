#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/afs_1/charles/codes/LightX2V_fastwam
PYTHON=${PYTHON:-/mnt/afs_1/charles/env/miniconda3/envs/lightx2v_libero_plus/bin/python}
EVALUATOR="$ROOT/lightx2v_train/tools/eval_fastwam_libero_checkpoint.py"
OUTPUT_ROOT=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_1step_30k
MODEL_PATH=/mnt/afs_1/charles/models/Wan2.2-TI2V-5B
POLICY_CONFIG="$ROOT/configs/fastwam/libero_plus_i2va_dmd_1step.json"
DATASET_STATS=/mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224_dataset_stats.json
LIBERO_ROOT=/mnt/afs_1/charles/codes/LIBERO-plus
SCRIPT_ARGS=("$@")

evaluate_adapter() {
    local label=$1
    local adapter=$2
    "$PYTHON" "$EVALUATOR" \
        --adapter "$adapter" \
        --output-root "$OUTPUT_ROOT/$label" \
        --model-path "$MODEL_PATH" \
        --policy-config "$POLICY_CONFIG" \
        --dataset-stats "$DATASET_STATS" \
        --libero-root "$LIBERO_ROOT" \
        --devices 1 2 3 4 5 6 7 \
        --episodes-per-task 50 \
        --tasks-per-shard 10 \
        --seed 0 \
        "${SCRIPT_ARGS[@]}"
}

evaluate_adapter \
    native \
    /mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt
evaluate_adapter \
    old_success_baseline_30k \
    "$ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_16gpu_mbs48_nogc/exports/checkpoint-000030000-student.pt"
evaluate_adapter \
    lora_only_30k \
    "$ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_only/exports/checkpoint-000030000-student.pt"

joint_adapter="$ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_joint/exports/checkpoint-000030000-student.pt"
if [[ -f "$joint_adapter" ]]; then
    rm -f "$OUTPUT_ROOT/joint_30k/PENDING.txt"
    evaluate_adapter joint_30k "$joint_adapter"
else
    mkdir -p "$OUTPUT_ROOT/joint_30k"
    printf 'pending: adapter has not been generated\nabsolute_adapter_path: %s\nchecked_at_utc: %s\n' \
        "$joint_adapter" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$OUTPUT_ROOT/joint_30k/PENDING.txt"
fi
