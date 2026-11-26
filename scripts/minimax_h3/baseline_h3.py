"""Run the single-GPU H3-20 baseline with the current LightX2V pipeline API."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A cinematic starship flies through a colorful nebula, engines humming.")
    parser.add_argument("--image", type=Path, help="First frame; selects I2AV (exercises vision patch Linear).")
    parser.add_argument("--last-frame", type=Path, help="Last frame with --image selects FL2AV.")
    parser.add_argument("--output", type=Path, default=ROOT / "save_results/h3_20/baseline.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=243)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--warmup", action="store_true", help="Run an excluded full-size request before measurement.")
    parser.add_argument("--check", action="store_true", help="Validate paths/imports, save config, exit without loading weights.")
    args = parser.parse_args()
    if args.last_frame and not args.image:
        parser.error("--last-frame requires --image")
    return args


def configure_environment():
    defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29680",
        "DTYPE": "BF16",
        "SENSITIVE_LAYER_DTYPE": "FP32",
        "PLATFORM": "cuda",
        "PROFILING_DEBUG_LEVEL": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    if int(os.environ["WORLD_SIZE"]) != 1:
        raise ValueError("baseline_h3.sh is a single-GPU baseline; do not launch it with torchrun.")
    # Source-built CUDA extensions from the validated 5090 environment (optional).
    runtime = Path(os.environ.get("H3_RUNTIME_ROOT", "/mnt/afs_1/charles/experiments/minimax_h3"))
    paths = [ROOT, ROOT / "vendor/lxk_clean", ROOT / "vendor/sparge_attn_cu130_current_stream", runtime / "report_20261021/runtime", runtime / "research_20261021/flashinfer031_runtime"]
    paths.extend(Path(p) for p in os.environ.get("H3_EXTRA_PYTHONPATH", "").split(os.pathsep) if p)
    sys.path[:0] = [str(p) for p in paths if p.is_dir()]


def make_config(args):
    models = Path(os.environ.get("H3_MODELS_DIR", "/mnt/afs_1/charles/models"))
    quant = Path(os.environ.get("H3_QUANT_DIR", str(models / "MiniMax-H3-quant")))
    return {
        "model_path": os.environ.get("H3_MODEL_ROOT", str(models / "MiniMax-H3-root")),
        "model_cls": "minimax_h3",
        "model_variant": "fl2av",
        "infer_steps": 4,
        "scheduler_type": "euler",
        "video_flow_shift": 6.0,
        "audio_flow_shift": 3.0,
        "enable_cfg": False,
        "cpu_offload": True,
        "offload_granularity": "block",
        "dit_prepost_resident": True,
        "use_adaln_cache": True,
        "adaln_cache_dir": os.environ.get("H3_ADALN_CACHE", str(ROOT / "save_results/h3_20/adaln_cache")),
        "dit_quantized": True,
        "dit_quant_scheme": "fp8-f16-accum",
        "dit_quantized_ckpt": os.environ.get("H3_DIT_CKPT", str(quant / "minimax_h3_dit_sla_merged_fp8_f16_accum.safetensors")),
        "text_encoder_cpu_offload": True,
        "text_encoder_offload_granularity": "block",
        "text_encoder_quantized": True,
        "text_encoder_quant_scheme": "fp8-sgl",
        "text_encoder_quantized_ckpt": os.environ.get("H3_TEXT_CKPT", str(quant / "minimax_h3_text_encoder_fp8_sgl.safetensors")),
        "video_vae_quantized": True,
        "video_vae_quant_scheme": "fp8-f16-accum",
        "video_vae_quantized_ckpt": os.environ.get("H3_VIDEO_VAE_CKPT", str(quant / "minimax_h3_video_vae_fp8_f16_accum_encoder_conv_q21.safetensors")),
        "vae_encoder_conv_mode": "cutlass_fp8_f16_accum",
        "vae_cpu_offload": False,
        "vae_sensitive_layer_dtype": "fp32",
        "vae_attn_type": "sage_attn2",
        "vae_use_compile": True,
        "vae_encode_parallel": True,
        "vae_decode_parallel": True,
        "vae_decode_tile_shape": {"768x1344": [304, 320], "544x960": [288, 304]},
        "attn_type": "dynamic_sparse_attn",
        "dynamic_sparse_attn_setting": {"sparsity_ratio": 0.9, "operator": "sage2"},
        "rms_type": "sgl-kernel",
        "rope_type": "torch_real_rope",
        "feature_caching": "NoCaching",
        "use_compile": True,
        "warmup": False,
        "changing_resolution": False,
        "use_fused_qkv": True,
        "lora_dynamic_apply": False,
        "seq_parallel": False,
        "parallel": {"seq_p_size": 1},
        "research_vision_patch_linear": True,
        "research_lightvae_decoder": os.environ.get("H3_LIGHTVAE_CKPT", str(models / "LightVAE/lightvae-lite-h3-decoder.safetensors")),
        "request_defaults": {"size": [args.height, args.width], "num_frames": args.frames},
        "video_codec_options": {"preset": "ultrafast", "crf": "23"},
    }


def main():
    args = parse_args()
    configure_environment()
    config = make_config(args)
    for key in ("model_path", "dit_quantized_ckpt", "text_encoder_quantized_ckpt", "video_vae_quantized_ckpt", "research_lightvae_decoder"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"{key}: {config[key]}")
    for path in (args.image, args.last_frame):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    # Fail early if required CUDA extensions are missing.
    import flashinfer  # noqa: F401
    import sageattention  # noqa: F401
    import spas_sage_attn  # noqa: F401
    import torch

    from lightx2v import LightX2VPipeline
    from lightx2v.common.ops.mm.fp8_f16_accum import fp8_f16_accum_mm_unavailable_reason
    from lightx2v.utils.set_config import build_startup_config
    from tools.h3_5090.patches import h3_lightvae_patch, h3_vision_linear_patch

    reason = fp8_f16_accum_mm_unavailable_reason()
    if reason is not None:
        raise RuntimeError(reason)
    config = dict(build_startup_config(config))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.check:
        print(f"H3-20 paths/imports/config OK: {output.with_suffix('.config.json')}")
        return
    from lightx2v.models.networks.minimax_h3.adaln_cache import resolve_adaln_cache
    from tools.cache_minimax_h3_adaln.builder import build_persistent_adaln_cache

    cache_path, _ = resolve_adaln_cache(config)
    if not cache_path.exists():
        build_persistent_adaln_cache(config)
    h3_lightvae_patch.install(config, output.parent)
    h3_vision_linear_patch.install(output.parent)
    request = {"task": "fl2av" if args.last_frame else "i2av" if args.image else "t2av", "prompt": args.prompt, "seed": args.seed, "size": [args.height, args.width], "num_frames": args.frames}
    if args.image:
        request["image_path"] = str(args.image.resolve())
    if args.last_frame:
        request["last_frame_path"] = str(args.last_frame.resolve())
    with LightX2VPipeline(config) as pipe:
        if args.warmup:
            pipe.generate(**request, save_result_path=str(output.with_name(output.stem + ".warmup.mp4")))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        pipe.generate(**request, save_result_path=str(output))
        torch.cuda.synchronize()
        metrics = {
            "baseline": "H3-20",
            "request": request,
            "output": str(output),
            "warmup": args.warmup,
            "request_s": time.perf_counter() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "triton", "safetensors", "av")},
        }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
