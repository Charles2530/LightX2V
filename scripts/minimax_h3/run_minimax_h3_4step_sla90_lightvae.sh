#!/bin/bash
# MiniMax-H3 single-GPU acceleration baseline:
#   4-step distilled weights + 90% sparse Sage2 SLA + vision patch Linear + LightVAE decoder.
# Config: configs/minimax_h3/distill/4steps/minimax_h3_fp8_4step_5090_sla90_lightvae.json
#
# Usage (extra arguments are passed to the Python entry, see --help):
#   bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup
#   bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup --image_path /path/to/first.png
#   CHANGING_RESOLUTION=1 bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup
#   TOKEN_SATURATION=1 bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup

# set path firstly
lightx2v_path=${lightx2v_path:-/path/to/LightX2V}
model_path=${model_path:-/path/to/MiniMax-H3}
quant_dir=${quant_dir:-/path/to/MiniMax-H3-quant}
lightvae_ckpt=${lightvae_ckpt:-/path/to/LightVAE/lightvae-lite-h3-decoder.safetensors}

export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export PROFILING_DEBUG_LEVEL=${PROFILING_DEBUG_LEVEL:-1}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Optional: locations of source-built CUDA extensions (sgl-kernel, SageAttention2, SpargeAttn, ...).
if [ -n "${EXTRA_PYTHONPATH}" ]; then
    export PYTHONPATH=${EXTRA_PYTHONPATH}:${PYTHONPATH}
fi

extra_args=()
if [ "${CHANGING_RESOLUTION:-0}" = "1" ]; then
    extra_args+=(--changing_resolution)
fi
if [ "${TOKEN_SATURATION:-0}" = "1" ]; then
    extra_args+=(--token_saturation)
fi

python "${lightx2v_path}/scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.py" \
  --model_path "${model_path}" \
  --dit_ckpt "${quant_dir}/minimax_h3_dit_sla_merged_fp8_f16_accum.safetensors" \
  --text_encoder_ckpt "${quant_dir}/minimax_h3_text_encoder_fp8_sgl.safetensors" \
  --video_vae_ckpt "${quant_dir}/minimax_h3_video_vae_fp8_f16_accum_encoder_conv_q21.safetensors" \
  --lightvae_ckpt "${lightvae_ckpt}" \
  --seed 0 \
  "${extra_args[@]}" \
  "$@"
