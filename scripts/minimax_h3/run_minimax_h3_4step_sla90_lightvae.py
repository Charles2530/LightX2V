"""MiniMax-H3 single-GPU acceleration baseline.

Stack: 4-step distilled (merged Turbo-SLA FP8 DiT) + 90% sparse Sage2 SLA +
vision patch Linear + LightVAE decoder. The model is configured by
``configs/minimax_h3/distill/4steps/minimax_h3_fp8_4step_5090_sla90_lightvae.json``;
``--changing_resolution`` turns on the progressive-resolution scheduler on top.

``--warmup`` runs the same request once before the timed request, so compile
and autotune for the real shape are excluded from ``request_s``. A missing
AdaLN cache is built in-process from the same config before loading.
"""

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/minimax_h3/distill/4steps/minimax_h3_fp8_4step_5090_sla90_lightvae.json"
DEFAULT_PROMPT = "A cinematic starship flies through a colorful nebula, engines humming."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="MiniMax-H3 model root")
    parser.add_argument("--config_json", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dit_ckpt", help="Override dit_quantized_ckpt")
    parser.add_argument("--text_encoder_ckpt", help="Override text_encoder_quantized_ckpt")
    parser.add_argument("--video_vae_ckpt", help="Override video_vae_quantized_ckpt")
    parser.add_argument("--lightvae_ckpt", help="Override video_vae_decoder_ckpt")
    parser.add_argument("--adaln_cache_dir", help="Override adaln_cache_dir")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image_path", help="First frame; selects I2AV")
    parser.add_argument("--last_frame_path", help="Last frame; with --image_path selects FL2AV")
    parser.add_argument("--save_result_path", default=str(REPO_ROOT / "save_results/minimax_h3_4step_sla90_lightvae.mp4"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num_frames", type=int, default=243)
    parser.add_argument("--changing_resolution", action="store_true", help="Enable progressive resolution (keys from the config)")
    parser.add_argument("--warmup", action="store_true", help="Run the request once before timing it")
    args = parser.parse_args()
    if args.last_frame_path and not args.image_path:
        parser.error("--last_frame_path requires --image_path")
    return args


def build_config(args):
    config = json.loads(args.config_json.read_text())
    overrides = {
        "dit_quantized_ckpt": args.dit_ckpt,
        "text_encoder_quantized_ckpt": args.text_encoder_ckpt,
        "video_vae_quantized_ckpt": args.video_vae_ckpt,
        "video_vae_decoder_ckpt": args.lightvae_ckpt,
        "adaln_cache_dir": args.adaln_cache_dir,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.update({"model_path": args.model_path, "model_cls": "minimax_h3", "model_variant": "fl2av"})
    if args.changing_resolution:
        config["changing_resolution"] = True
    return config


def ensure_adaln_cache(config):
    from lightx2v.models.networks.minimax_h3.adaln_cache import resolve_adaln_cache
    from lightx2v.utils.set_config import build_startup_config
    from tools.cache_minimax_h3_adaln.builder import build_persistent_adaln_cache

    startup = build_startup_config(config)
    cache_path, _ = resolve_adaln_cache(startup)
    if not cache_path.exists():
        build_persistent_adaln_cache(startup)


def main():
    args = parse_args()
    config = build_config(args)

    import torch

    from lightx2v import LightX2VPipeline

    if config.get("use_adaln_cache", False):
        ensure_adaln_cache(config)

    task = "fl2av" if args.last_frame_path else "i2av" if args.image_path else "t2av"
    request = {"task": task, "prompt": args.prompt, "seed": args.seed, "size": [args.height, args.width], "num_frames": args.num_frames}
    if args.image_path:
        request["image_path"] = args.image_path
    if args.last_frame_path:
        request["last_frame_path"] = args.last_frame_path
    output = Path(args.save_result_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with LightX2VPipeline(config) as pipe:
        if args.warmup:
            pipe.generate(**request, save_result_path=str(output.with_name(f"{output.stem}.warmup{output.suffix}")))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        pipe.generate(**request, save_result_path=str(output))
        torch.cuda.synchronize()
        request_s = time.perf_counter() - start
    summary = {
        "task": task,
        "changing_resolution": bool(config.get("changing_resolution")),
        "warmup": args.warmup,
        "request_s": round(request_s, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
