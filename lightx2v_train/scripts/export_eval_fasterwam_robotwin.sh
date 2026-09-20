#!/usr/bin/env bash
# Separate parameterized entry: leave historical evaluation scripts unchanged.
set -eo pipefail
if (( $# < 1 )) || [[ "$1" == --help || "$1" == -h ]]; then
  printf 'Usage: bash %s CHECKPOINT_DIR [Hydra overrides...]\n' "$0"
  if (( $# < 1 )); then exit 2; fi
  exit 0
fi
checkpoint_arg=$1
shift
if [[ ! -d "$checkpoint_arg" || ! -f "$checkpoint_arg/config.yaml" || ! -f "$checkpoint_arg/ema_action.pt" ]]; then
  printf 'Expected a consistency checkpoint directory containing config.yaml and ema_action.pt: %s\n' "$checkpoint_arg" >&2
  exit 2
fi
ckpt=$(cd "$checkpoint_arg" && pwd)
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
export FASTWAM_ROOT=${FASTWAM_ROOT:-/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM}
export FASTERWAM_ROOT=${FASTERWAM_ROOT:-/mnt/miaohua/charles/codes/FasterWAM}
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/miaohua/charles/models/fastWAM-compat
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_SAMPLER=flow_matching
export PYTHONPATH="$repo/lightx2v_train:$repo:$FASTERWAM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${OUT:-}" ]]; then
  mkdir -p "$repo/evaluate_results/robotwin"
  OUT=$(mktemp -d "$repo/evaluate_results/robotwin/fasterwam_$(basename "$(dirname "$ckpt")")_$(basename "$ckpt")_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
mkdir -p "$OUT"
export OUT=$(cd "$OUT" && pwd)
weight="$OUT/export/ema_merged.pt"
lora="$OUT/export/lora"
# Reuse is explicit and read-only, for resizing an existing evaluation.
if [[ "${REUSE_EXPORTED_EMA:-0}" == 1 ]]; then
  test -s "$weight"
  test -s "$lora/action/adapter_model.safetensors"
  "$repo/.venv/bin/python" - "$ckpt/ema_action.pt" "$lora/action/adapter_model.safetensors" <<'PY'
import sys
import torch
from safetensors.torch import load_file
source = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
exported = load_file(sys.argv[2], device='cpu')
assert set(source) == set(exported), 'EMA adapter keys do not match checkpoint'
assert all(torch.equal(source[key], exported[key]) for key in source), 'EMA adapter differs from checkpoint'
print('Reusing existing export: EMA adapter exactly matches the requested checkpoint', flush=True)
PY
else
# Never overwrite exports that another evaluation could be reading.
if [[ -e "$OUT/export" ]]; then
  printf 'Export directory already exists; choose a new OUT: %s/export\n' "$OUT" >&2
  exit 2
fi
mkdir "$OUT/export"
CUDA_VISIBLE_DEVICES=${EXPORT_GPU:-0} "$repo/.venv/bin/python" -u \
  lightx2v_train/tools/export_fastwam_action_dmd.py \
  --config "$ckpt/config.yaml" --checkpoint "$ckpt" --weights ema \
    --lora-output "$lora" --output "$weight" 2>&1 | tee "$OUT/export.log"
fi
test -s "$weight"
test -s "$lora/action/adapter_model.safetensors"
test -s "$lora/action/adapter_config.json"
printf 'Checkpoint: %s\nMerged EMA: %s\nLoRA: %s/action\nResults: %s\n' "$ckpt" "$weight" "$lora" "$OUT"

source /mnt/miaohua/charles/envs/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
set -u
python="$CONDA_PREFIX/bin/python"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
# Allow independent evaluations to use disjoint GPUs; preserve historical default.
export CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
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
cd "${EVAL_RUNTIME_ROOT:-$FASTWAM_ROOT}"
export PYTHONPATH="$FASTWAM_ROOT/src:$FASTWAM_ROOT:$PYTHONPATH"
"$python" -u "$repo/lightx2v_train/tools/run_robotwin_fasterwam_eval.py" \
  ckpt="$weight" task=robotwin_fasterwam_future_kv_1step \
  EVALUATION.robotwin_root="$FASTWAM_ROOT/third_party/RoboTwin" \
  EVALUATION.dataset_stats_path=/mnt/miaohua/charles/models/fasterwam_release/robotwin/dataset_stats.json \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.num_inference_steps=1 EVALUATION.sigma_shift=5.0 \
  EVALUATION.replan_steps=28 EVALUATION.instruction_type=unseen \
  EVALUATION.skip_get_obs_within_replan=true EVALUATION.timing_enabled=true \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 \
  EVALUATION.output_dir="$OUT" "$@" 2>&1 | tee -a "$OUT/launcher.log"
