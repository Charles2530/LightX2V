#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
mode=${1:-full}
if [[ "$mode" == smoke || "$mode" == full ]]; then
  if (( $# )); then shift; fi
else
  printf 'Usage: bash %s [full|smoke] [Hydra overrides...]\n' "$0" >&2
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
if [[ ! -s "$weight" || ! -s "$lora/action/adapter_model.safetensors" ]]; then
  CUDA_VISIBLE_DEVICES=${EXPORT_GPU:-0} "$repo/.venv/bin/python" -u \
    lightx2v_train/tools/export_fastwam_action_dmd.py \
    --config "$ckpt/config.yaml" --checkpoint "$ckpt" --weights ema \
    --lora-output "$lora" --output "$weight"
fi
episodes=100
if [[ "$mode" == smoke ]]; then
  episodes=1
  export ROBOTWIN_MANAGER_ENTRY="$repo/lightx2v_train/tools/run_robotwin_fasterwam_smoke.py"
  export ROBOTWIN_SMOKE_TASK_LIMIT=16
else
  unset ROBOTWIN_MANAGER_ENTRY
fi
export OUT=${OUT:-$FASTWAM_ROOT/evaluate_results/robotwin/fasterwam_robotwin_consistency_ts10_ema_step30000/${mode}_$(date +%Y%m%d_%H%M%S)}
bash lightx2v_train/scripts/eval_robotwin_distilled_1step.sh consistency \
  ckpt="$weight" task=robotwin_uncond_3cam_384_distilled_1step_fasterwam \
  EVALUATION.robotwin_root="$FASTWAM_ROOT/third_party/RoboTwin" \
  EVALUATION.eval_num_episodes="$episodes" \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 \
  EVALUATION.output_dir="$OUT" "$@"
