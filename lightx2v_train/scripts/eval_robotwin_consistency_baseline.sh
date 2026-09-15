#!/usr/bin/env bash
# Explicit sampler selection; keep the existing evaluation launcher unchanged.
set -eo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash eval_robotwin_consistency_baseline.sh --sampler consistency_baseline|flow_matching [--sigma-data 0.5] ckpt=/path/to/merged.pt [Hydra overrides...]' \
    'Set OUT to the result directory. Both sampler variables are scoped to this launcher.'
}

sampler=
sigma_data=0.5
while (( $# )); do
  case "$1" in
    --sampler|--sigma-data)
      if (( $# < 2 )); then usage >&2; exit 2; fi
      if [[ "$1" == --sampler ]]; then sampler=$2; else sigma_data=$2; fi
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    --*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done
case "$sampler" in
  consistency_baseline|flow_matching) ;;
  *) printf 'Pass --sampler consistency_baseline or --sampler flow_matching.\n' >&2; exit 2 ;;
esac

has_checkpoint=false
for override in "$@"; do
  if [[ "$override" == ckpt=?* ]]; then has_checkpoint=true; fi
done
if [[ "$has_checkpoint" != true ]]; then
  printf 'An explicit ckpt=/path/to/merged.pt override is required.\n' >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FASTWAM_ACTION_SAMPLER="$sampler"
export FASTWAM_CONSISTENCY_SIGMA_DATA="$sigma_data"
export OUT=${OUT:-/mnt/miaohua/charles/codes/FastWAM/evaluate_results/robotwin/robotwin_${sampler}_1step_$(date +%Y%m%d_%H%M%S)}
exec bash "$script_dir/eval_robotwin_distilled_1step.sh" consistency "$@"
