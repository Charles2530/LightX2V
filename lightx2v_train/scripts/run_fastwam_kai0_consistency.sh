#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}/lightx2v_train"
export PYTHONPATH="${repo_root}:${repo_root}/lightx2v_train:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
python=${PYTHON:-${repo_root}/.venv/bin/python}
mode=${CONSISTENCY_MODE:-action}
case "${mode}" in
  action) default_config=configs/train/fastwam_action_dmd/kai0_flatten_fold_action_1step_consistency.yaml ;;
  joint) default_config=configs/train/fastwam_action_dmd/kai0_flatten_fold_action_1step_consistency_joint.yaml ;;
  *) echo "CONSISTENCY_MODE must be action or joint" >&2; exit 2 ;;
esac
config=${CONFIG:-${default_config}}
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the trained Kai0 checkpoint fastwam.pt}"
if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "Teacher checkpoint does not exist: ${TEACHER_CHECKPOINT}" >&2
  exit 2
fi
export TEACHER_CHECKPOINT

nnodes=${NNODES:-1}
launcher_args=(--nproc_per_node="${NPROC_PER_NODE:-8}")
if [[ "${nnodes}" == "1" ]]; then
  launcher_args+=(--standalone)
else
  : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node IP or hostname}"
  : "${NODE_RANK:?Set NODE_RANK to 0 through NNODES-1 on each node}"
  launcher_args+=(
    --nnodes="${nnodes}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT:-29500}"
  )
fi

exec "${python}" -m torch.distributed.run \
  "${launcher_args[@]}" \
  train.py --config "${config}"
