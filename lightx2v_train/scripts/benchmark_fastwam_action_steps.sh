#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
config=${CONFIG:-${lightx2v_path}/lightx2v_train/configs/train/fastwam_action_dmd/libero_action_1step_dmd_lora.yaml}

cd "${lightx2v_path}/lightx2v_train"
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

args=(
  --config "${config}"
  --steps "${STEPS:-1,2,4,5,10,20}"
  --num-samples "${NUM_SAMPLES:-16}"
  --seed "${SEED:-0}"
)
if [[ -n "${CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi

python tools/benchmark_fastwam_action_steps.py "${args[@]}"
