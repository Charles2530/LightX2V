#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/mnt/afs_1/charles/codes/LightX2V_fastwam}
config=${CONFIG:-${lightx2v_path}/lightx2v_train/configs/train/fastwam_action_dmd/libero_action_1step_dmd_lora_only.yaml}

nnodes=${NNODES:-2}
nproc_per_node=${NPROC_PER_NODE:-8}
master_addr=${MASTER_ADDR:?Set MASTER_ADDR to the master node IP or resolvable hostname}
master_port=${MASTER_PORT:-29500}
rdzv_backend=${RDZV_BACKEND:-c10d}
rdzv_id=${RDZV_ID:-fastwam-libero-action-dmd-16gpu}
python=${PYTHON:-${lightx2v_path}/.venv/bin/python}
: "${WANDB_API_KEY:?Set WANDB_API_KEY on both nodes before launching}"

cd "${lightx2v_path}/lightx2v_train"
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

"${python}" -m torch.distributed.run \
  --nnodes="${nnodes}" \
  --nproc_per_node="${nproc_per_node}" \
  --rdzv_backend="${rdzv_backend}" \
  --rdzv_endpoint="${master_addr}:${master_port}" \
  --rdzv_id="${rdzv_id}" \
  train.py --config "${config}"
