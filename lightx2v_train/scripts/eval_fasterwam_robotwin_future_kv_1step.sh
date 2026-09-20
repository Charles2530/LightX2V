#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
mode=${1:-full}
if [[ "$mode" == smoke || "$mode" == full || "$mode" == export ]]; then
  if (( $# )); then shift; fi
else
  printf 'Usage: bash %s [full|smoke|export] [Hydra overrides...]\n' "$0" >&2
  exit 2
fi
export FASTWAM_ROOT=${FASTWAM_ROOT:-/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM}
export FASTERWAM_ROOT=${FASTERWAM_ROOT:-/mnt/miaohua/charles/codes/FasterWAM}
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/miaohua/charles/models/fastWAM-compat
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_SAMPLER=flow_matching
export PYTHONPATH="$repo/lightx2v_train:$repo:$FASTERWAM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ckpt="$repo/lightx2v_train/runs/fasterwam_robotwin_action_1step_consistency_ts10/checkpoint-000030000"
weight="$repo/fasterwam_robotwin_consistency_ts10_ema_step30000.pt"
lora="$repo/fasterwam_robotwin_consistency_ts10_ema_step30000_lora"
if [[ "$mode" == export || ! -s "$weight" || ! -s "$lora/action/adapter_model.safetensors" || ! -s "$lora/action/adapter_config.json" ]]; then
  CUDA_VISIBLE_DEVICES=${EXPORT_GPU:-0} "$repo/.venv/bin/python" -u \
    lightx2v_train/tools/export_fastwam_action_dmd.py \
    --config "$ckpt/config.yaml" --checkpoint "$ckpt" --weights ema \
    --lora-output "$lora" --output "$weight"
fi
printf 'EMA checkpoint: %s\nLoRA adapter: %s/action\n' "$weight" "$lora"
if [[ "$mode" == export ]]; then
  exit 0
fi

# Independent FasterWAM launcher. Do not inherit FastWAM inference defaults.
source /mnt/miaohua/charles/envs/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
set -u
python="$CONDA_PREFIX/bin/python"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUROBO_TORCH_COMPILE_DISABLE=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
unset ROBOTWIN_FORCE_RASTER LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE
: "${ROBOTWIN_NVIDIA_GL_ROOT:?RoboTwin activation must configure ROBOTWIN_NVIDIA_GL_ROOT}"
export VK_ICD_FILENAMES="$ROBOTWIN_NVIDIA_GL_ROOT/nvidia_icd_abs.json"
export LD_PRELOAD="$ROBOTWIN_NVIDIA_GL_ROOT/libGL.so.1.7.0"
"$python" - <<'PY'
from importlib.metadata import version
actual = version("sapien")
if actual != "3.0.0b1":
    raise SystemExit(f"FasterWAM RoboTwin evaluation requires sapien==3.0.0b1, found {actual}.")
PY

episodes=100
manager="$FASTWAM_ROOT/experiments/robotwin/run_robotwin_manager.py"
if [[ "$mode" == smoke ]]; then
  episodes=1
  manager="$repo/lightx2v_train/tools/run_robotwin_fasterwam_smoke.py"
  export ROBOTWIN_SMOKE_TASK_LIMIT=${ROBOTWIN_SMOKE_TASK_LIMIT:-16}
fi
export OUT=${OUT:-$FASTWAM_ROOT/evaluate_results/robotwin/fasterwam_robotwin_consistency_ts10_ema_step30000/${mode}_future_kv_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
cd "$FASTWAM_ROOT"
export PYTHONPATH="$FASTWAM_ROOT/src:$FASTWAM_ROOT:$PYTHONPATH"
"$python" -u "$manager" \
  ckpt="$weight" task=robotwin_fasterwam_future_kv_1step \
  EVALUATION.robotwin_root="$FASTWAM_ROOT/third_party/RoboTwin" \
  EVALUATION.dataset_stats_path=/mnt/miaohua/charles/models/fasterwam_release/robotwin/dataset_stats.json \
  EVALUATION.eval_num_episodes="$episodes" \
  EVALUATION.num_inference_steps=1 EVALUATION.sigma_shift=5.0 \
  EVALUATION.replan_steps=28 EVALUATION.instruction_type=unseen \
  EVALUATION.skip_get_obs_within_replan=true EVALUATION.timing_enabled=true \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 \
  EVALUATION.output_dir="$OUT" "$@" 2>&1 | tee "$OUT/launcher.log"
