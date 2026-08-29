#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
config=${CONFIG:-${lightx2v_path}/lightx2v_train/configs/train/fastwam_action_dmd/libero_action_1step_dmd_lora.yaml}
checkpoint=${CHECKPOINT:?Set CHECKPOINT to a FastWAM action DMD checkpoint directory}
output=${OUTPUT:?Set OUTPUT to the exported .pt path}

cd "${lightx2v_path}/lightx2v_train"
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"

python=${PYTHON:-${lightx2v_path}/.venv/bin/python}

"${python}" tools/export_fastwam_action_dmd.py \
  --config "${config}" \
  --checkpoint "${checkpoint}" \
  --output "${output}"
