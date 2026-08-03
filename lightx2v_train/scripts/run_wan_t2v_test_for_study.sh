#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CONFIG="${CONFIG:-configs/train/flow/wan2_1_t2v_1_3b_full_for_study.yaml}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    train.py --config "${CONFIG}"
