#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
config=${CONFIG:-${lightx2v_path}/lightx2v_train/configs/train/fastwam_action_dmd/libero_action_1step_dmd_lora.yaml}

cd "${lightx2v_path}/lightx2v_train"
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

nproc_per_node=${NPROC_PER_NODE:-8}
python=${PYTHON:-${lightx2v_path}/.venv/bin/python}

"${python}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${nproc_per_node}" \
  train.py --config "${config}"
