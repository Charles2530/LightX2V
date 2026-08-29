#!/usr/bin/env bash
set -euo pipefail

# Spawned LIBERO environments otherwise inherit host-wide 96/192-thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/mnt/afs_1/charles/codes/LightX2V_fastwam}
EVAL_ROOT=${EVAL_ROOT:-$WORKSPACE_ROOT}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$WORKSPACE_ROOT}
PYTHON=${PYTHON:-/mnt/afs_1/charles/env/miniconda3/envs/lightx2v_libero_plus/bin/python}
EVALUATOR="$EVAL_ROOT/lightx2v_train/tools/eval_fastwam_libero_shared_checkpoint.py"
AGGREGATOR="$EVAL_ROOT/lightx2v_train/tools/aggregate_fastwam_libero_shared_results.py"
OUTPUT_ROOT=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_1step_30k
MODEL_PATH=/mnt/afs_1/charles/models/Wan2.2-TI2V-5B
POLICY_CONFIG="$EVAL_ROOT/configs/fastwam/libero_plus_i2va_dmd_1step.json"
DATASET_STATS=/mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224_dataset_stats.json
LIBERO_ROOT=/mnt/afs_1/charles/codes/LIBERO-plus
NVIDIA_EGL_ROOT=${NVIDIA_EGL_ROOT:-/mnt/afs_1/charles/env/nvidia-egl-550.90.07/root}
ENV_WORKERS_PER_DEVICE=${ENV_WORKERS_PER_DEVICE:-12}
PROTOCOL_DIRECTORY=${PROTOCOL_DIRECTORY:-official_protocol_shared_policy}
SCRIPT_ARGS=("$@")

evaluate_adapter() {
    local label=$1
    local adapter=$2
    "$PYTHON" "$EVALUATOR" \
        --adapter "$adapter" \
        --output-root "$OUTPUT_ROOT/$label/$PROTOCOL_DIRECTORY" \
        --model-path "$MODEL_PATH" \
        --policy-config "$POLICY_CONFIG" \
        --dataset-stats "$DATASET_STATS" \
        --libero-root "$LIBERO_ROOT" \
        --devices 1 2 3 4 5 6 7 \
        --env-workers-per-device "$ENV_WORKERS_PER_DEVICE" \
        --episodes-per-task 50 \
        --tasks-per-shard 1 \
        --seed 0 \
        --expected-action-infer-steps 1 \
        --expected-actions-per-plan 10 \
        --prompt-cache-limit 256 \
        --startup-timeout 3600 \
        --nvidia-egl-root "$NVIDIA_EGL_ROOT" \
        "${SCRIPT_ARGS[@]}"
}

evaluate_adapter \
    native \
    /mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt
evaluate_adapter \
    old_success_baseline_30k \
    "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_16gpu_mbs48_nogc/exports/checkpoint-000030000-student.pt"
evaluate_adapter \
    lora_only_30k \
    "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_only/exports/checkpoint-000030000-student.pt"

joint_adapter="$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_joint/exports/checkpoint-000030000-student.pt"
if [[ -f "$joint_adapter" ]]; then
    rm -f "$OUTPUT_ROOT/joint_30k/PENDING.txt"
    evaluate_adapter joint_30k "$joint_adapter"
else
    mkdir -p "$OUTPUT_ROOT/joint_30k"
    printf 'pending: adapter has not been generated\nabsolute_adapter_path: %s\nchecked_at_utc: %s\n' \
        "$joint_adapter" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$OUTPUT_ROOT/joint_30k/PENDING.txt"
fi

if [[ -f "$OUTPUT_ROOT/joint_30k/$PROTOCOL_DIRECTORY/summary.json" ]]; then
    "$PYTHON" "$AGGREGATOR" \
        --results-root "$OUTPUT_ROOT" \
        --protocol-directory "$PROTOCOL_DIRECTORY" \
        --weight native native /mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt \
        --weight old_success_baseline_30k old_success_baseline_30k \
            "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_16gpu_mbs48_nogc/exports/checkpoint-000030000-student.pt" \
        --weight lora_only_30k lora_only_30k \
            "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_only/exports/checkpoint-000030000-student.pt" \
        --weight joint_30k joint_30k \
            "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_joint/exports/checkpoint-000030000-student.pt"
fi
