#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs_1/charles/codes/LightX2V_fastwam"
TRAIN_ROOT="${ROOT}/lightx2v_train"
EVAL_ROOT="${TRAIN_ROOT}/runs/fastwam_robotwin_action_1step_dmd_lora_only/robotwin_eval"
LOG_DIR="${EVAL_ROOT}/logs"
PID_FILE="${EVAL_ROOT}/watcher.pid"
STDOUT_LOG="${LOG_DIR}/watcher.stdout.log"

mkdir -p "${LOG_DIR}"

cd "${ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${TRAIN_ROOT}:${PYTHONPATH:-}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"

exec "${TRAIN_ROOT}/../.venv/bin/python" -u \
  "${TRAIN_ROOT}/tools/watch_fastwam_robotwin_eval.py" \
  --watch \
  --run-baseline \
  --step-interval 2000 \
  --max-step 30000 \
  --eval-num-episodes 1 \
  --baseline-num-episodes 100 \
  --num-gpus "${ROBOTWIN_EVAL_NUM_GPUS:-8}" \
  --max-tasks-per-gpu "${ROBOTWIN_EVAL_MAX_TASKS_PER_GPU:-1}" \
  --poll-seconds "${ROBOTWIN_EVAL_POLL_SECONDS:-60}" \
  --report-seconds "${ROBOTWIN_EVAL_REPORT_SECONDS:-1800}" \
  --render-retry-seconds "${ROBOTWIN_EVAL_RENDER_RETRY_SECONDS:-1800}" \
  --nvidia-visible-devices "${ROBOTWIN_EVAL_NVIDIA_VISIBLE_DEVICES:-all}"
