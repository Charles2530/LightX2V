#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/afs_1/charles/env/miniconda3/envs/robotwin/bin/python}"
FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/afs_1/charles/codes/FastWAM}"
MANAGER="${MANAGER:-${FASTWAM_ROOT}/experiments/robotwin/run_robotwin_manager.py}"
SIM_CONFIG="${SIM_CONFIG:-${FASTWAM_ROOT}/configs/sim_robotwin.yaml}"
TASK="${TASK:-robotwin_uncond_3cam_384_1e-4}"
TASK_CONFIG="${TASK_CONFIG:-${FASTWAM_ROOT}/configs/task/${TASK}.yaml}"
WEIGHT="${WEIGHT:-/mnt/afs_1/charles/models/fastwam/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/mnt/afs_1/charles/models/fastwam/robotwin_uncond_3cam_384_dataset_stats.json}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs_1/charles/models/Wan2.2-TI2V-5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/afs_1/charles/models/Wan2.1-T2V-1.3B}"
EPISODES_PER_PHASE="${EPISODES_PER_PHASE:-100}"
ACTION_INFER_STEPS="${ACTION_INFER_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_ID="${RUN_ID:-robotwin_$(date -u +%Y%m%dT%H%M%SZ)}"
WEIGHT_TAG="$(basename -- "${WEIGHT%.*}")"
OUTPUT_DIR="${OUTPUT_DIR:-${FASTWAM_ROOT}/evaluate_results/robotwin/${WEIGHT_TAG}/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/launcher.log}"
DRY_RUN="${DRY_RUN:-0}"

RENDER_ENV_ROOT="${RENDER_ENV_ROOT:-${ROOT}/lightx2v_train/runs/fastwam_robotwin_action_1step_dmd_lora_only/robotwin_eval}"
NVIDIA_DRIVER_LIB_DIR="${NVIDIA_DRIVER_LIB_DIR:-${RENDER_ENV_ROOT}/nvidia_gl_550_extracted/usr/lib/x86_64-linux-gnu}"
VULKAN_LOADER_LIB_DIR="${VULKAN_LOADER_LIB_DIR:-${RENDER_ENV_ROOT}/mesa_vulkan_extracted/usr/lib/x86_64-linux-gnu}"
if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
    if [[ -f "${RENDER_ENV_ROOT}/nvidia_gl_550_extracted/nvidia_icd_abs.json" ]]; then
        VK_ICD_FILENAMES="${RENDER_ENV_ROOT}/nvidia_gl_550_extracted/nvidia_icd_abs.json"
    else
        VK_ICD_FILENAMES="$(dirname -- "$(dirname -- "$PYTHON")")/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json"
    fi
fi
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    __EGL_VENDOR_LIBRARY_FILENAMES="${RENDER_ENV_ROOT}/nvidia_gl_550_extracted/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
fi
__GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
NVIDIA_VISIBLE_DEVICES="${ROBOTWIN_NVIDIA_VISIBLE_DEVICES:-all}"

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
require_dir "FastWAM project" "$FASTWAM_ROOT"
require_file "manager" "$MANAGER"
require_file "simulation config" "$SIM_CONFIG"
require_file "task config" "$TASK_CONFIG"
require_file "weight" "$WEIGHT"
require_file "dataset stats" "$DATASET_STATS"
require_dir "base model" "$MODEL_PATH"
require_dir "tokenizer" "$TOKENIZER_PATH"
require_file "NVIDIA Vulkan ICD" "$VK_ICD_FILENAMES"
require_file "NVIDIA EGL vendor config" "$__EGL_VENDOR_LIBRARY_FILENAMES"
require_dir "NVIDIA driver library" "$NVIDIA_DRIVER_LIB_DIR"
require_dir "Vulkan loader library" "$VULKAN_LOADER_LIB_DIR"
require_positive_int "EPISODES_PER_PHASE" "$EPISODES_PER_PHASE"
require_positive_int "ACTION_INFER_STEPS" "$ACTION_INFER_STEPS"
require_positive_int "REPLAN_STEPS" "$REPLAN_STEPS"
require_positive_int "MAX_TASKS_PER_GPU" "$MAX_TASKS_PER_GPU"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1: $DRY_RUN"
read -r -a DEVICES <<< "$GPU_IDS"
[[ "${#DEVICES[@]}" -gt 0 ]] || fail "GPU_IDS must contain at least one device"
for device in "${DEVICES[@]}"; do
    [[ "$device" =~ ^[0-9]+$ ]] || fail "GPU_IDS must contain non-negative integers: $GPU_IDS"
done
NUM_GPUS="${NUM_GPUS:-${#DEVICES[@]}}"
require_positive_int "NUM_GPUS" "$NUM_GPUS"
[[ "$NUM_GPUS" -eq "${#DEVICES[@]}" ]] || fail "NUM_GPUS must match the number of GPU_IDS entries"
GPU_IDS_HYDRA="$(IFS=,; printf '%s' "${DEVICES[*]}")"

mkdir -p -- "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export NVIDIA_VISIBLE_DEVICES
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export VK_ICD_FILENAMES
export __EGL_VENDOR_LIBRARY_FILENAMES
export __GLX_VENDOR_LIBRARY_NAME
export LD_LIBRARY_PATH="${VULKAN_LOADER_LIB_DIR}:${NVIDIA_DRIVER_LIB_DIR}:$(dirname -- "$(dirname -- "$PYTHON")")/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "$PYTHON" -u "$MANAGER"
    "task=$TASK"
    "ckpt=$WEIGHT"
    "EVALUATION.dataset_stats_path=$DATASET_STATS"
    "EVALUATION.eval_num_episodes=$EPISODES_PER_PHASE"
    "EVALUATION.num_inference_steps=$ACTION_INFER_STEPS"
    "EVALUATION.replan_steps=$REPLAN_STEPS"
    "EVALUATION.instruction_type=unseen"
    "EVALUATION.skip_get_obs_within_replan=true"
    "EVALUATION.output_dir=$OUTPUT_DIR"
    "MULTIRUN.num_gpus=$NUM_GPUS"
    "+MULTIRUN.gpu_ids=[$GPU_IDS_HYDRA]"
    "MULTIRUN.max_tasks_per_gpu=$MAX_TASKS_PER_GPU"
    "model.model_id=$MODEL_PATH"
    "model.tokenizer_model_id=$TOKENIZER_PATH"
    "model.redirect_common_files=false"
    "model.skip_dit_load_from_pretrain=true"
)
COMMAND+=("$@")

printf 'Mode: %s\n' "$([[ "$DRY_RUN" == "1" ]] && printf DRY_RUN || printf EVALUATION)"
printf 'Repository root: %s\n' "$ROOT"
printf 'Output directory: %s\n' "$OUTPUT_DIR"
printf 'Environment:\n'
printf '  PYTHON=%s\n' "$PYTHON"
printf '  PYTHONPATH=%s\n' "$PYTHONPATH"
printf '  LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
printf '  NVIDIA_VISIBLE_DEVICES=%s\n' "$NVIDIA_VISIBLE_DEVICES"
printf '  GPU_IDS=%s\n' "$GPU_IDS"
printf '  VK_ICD_FILENAMES=%s\n' "$VK_ICD_FILENAMES"
printf '  __EGL_VENDOR_LIBRARY_FILENAMES=%s\n' "$__EGL_VENDOR_LIBRARY_FILENAMES"
printf '  __GLX_VENDOR_LIBRARY_NAME=%s\n' "$__GLX_VENDOR_LIBRARY_NAME"
printf 'Command:\n  '
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

cd -- "$FASTWAM_ROOT"
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
