import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch
from loguru import logger

from lightx2v import LightX2VPipeline


GIB = 1024**3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan2.1-I2V-14B with disk/CPU/GPU block offload."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--save-result-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gib", type=float, default=8.0)
    parser.add_argument("--allocator-budget-gib", type=float, default=7.5)
    parser.add_argument("--cpu-budget-gib", type=float, default=16.0)
    return parser.parse_args()


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_peaks():
    return {
        "allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }


def process_peak_rss_gib():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024 / GIB
    except (OSError, ValueError, IndexError):
        pass
    return None


def write_metrics(path, metrics):
    metrics_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(metrics_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)


def validate_three_level_offload(pipe):
    config = pipe.runner.config
    checks = {
        "cpu_offload": config.get("cpu_offload", False) is True,
        "block_granularity": config.get("offload_granularity") == "block",
        "lazy_load": config.get("lazy_load", False) is True,
        "lazy_model_deferred": getattr(pipe.runner, "model", None) is None,
        "t5_cpu_offload": config.get("t5_cpu_offload", False) is True,
        "clip_cpu_offload": config.get("clip_cpu_offload", False) is True,
        "vae_cpu_offload": config.get("vae_cpu_offload", False) is True,
        "dit_quantized": config.get("dit_quantized", False) is True,
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Three-level offload validation failed: " + ", ".join(failed)
        )

    logger.info(
        "[OFFLOAD/3LEVEL] configuration confirmed: "
        "lazy disk load -> CPU block buffers -> GPU block buffers"
    )
    logger.info(
        "[OFFLOAD/3LEVEL] model creation is correctly deferred until generation; "
        "runtime buffer creation and transfers will be reported by "
        "WeightAsyncStreamManager trace logs"
    )
    return checks


def configure_cuda_budget(allocator_budget_gib):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    total_bytes = device_properties.total_memory
    requested_bytes = allocator_budget_gib * GIB
    allocator_limit_bytes = min(requested_bytes, total_bytes * 0.95)
    memory_fraction = allocator_limit_bytes / total_bytes
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device_index)
    logger.info(
        "[MEMORY/BUDGET] device={}, physical_vram={:.3f} GiB, "
        "pytorch_allocator_limit={:.3f} GiB ({:.2%})",
        device_properties.name,
        total_bytes / GIB,
        allocator_limit_bytes / GIB,
        memory_fraction,
    )
    return {
        "device_name": device_properties.name,
        "physical_vram_gib": total_bytes / GIB,
        "allocator_limit_gib": allocator_limit_bytes / GIB,
        "allocator_fraction": memory_fraction,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.save_result_path)), exist_ok=True)

    metrics = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model": "Wan2.1-I2V-14B-480P",
        "resolution": [480, 832],
        "num_frames": 81,
        "gpu_budget_gib": args.gpu_budget_gib,
        "cpu_budget_gib": args.cpu_budget_gib,
        "config_json": os.path.abspath(args.config_json),
        "output_video": os.path.abspath(args.save_result_path),
    }
    overall_start = time.perf_counter()
    active_phase = "initialization"

    try:
        metrics["cuda_budget"] = configure_cuda_budget(args.allocator_budget_gib)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        init_start = time.perf_counter()
        pipe = LightX2VPipeline(
            model_path=args.model_path,
            model_cls="wan2.1",
            task="i2v",
        )
        pipe.enable_offload(
            cpu_offload=True,
            offload_granularity="block",
            text_encoder_offload=True,
            image_encoder_offload=True,
            vae_offload=True,
        )
        pipe.create_generator(config_json=args.config_json)
        synchronize_cuda()
        metrics["initialization"] = {
            "elapsed_seconds": time.perf_counter() - init_start,
            "cuda_peak": cuda_peaks(),
        }
        metrics["offload_validation"] = validate_three_level_offload(pipe)

        active_phase = "generation"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        generation_start = time.perf_counter()
        pipe.generate(
            seed=args.seed,
            image_path=args.image_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            save_result_path=args.save_result_path,
        )
        synchronize_cuda()
        metrics["generation"] = {
            "elapsed_seconds": time.perf_counter() - generation_start,
            "cuda_peak": cuda_peaks(),
        }
        metrics["status"] = "success"
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            synchronize_cuda()
            current_peaks = cuda_peaks() if torch.cuda.is_available() else None
            phase_metrics = metrics.setdefault(active_phase, {})
            phase_metrics.setdefault("cuda_peak", current_peaks)
        except Exception as sync_exc:
            metrics["final_sync_error"] = f"{type(sync_exc).__name__}: {sync_exc}"

        metrics["total_elapsed_seconds"] = time.perf_counter() - overall_start
        metrics["cpu_peak_rss_gib"] = process_peak_rss_gib()
        peak_records = [
            section.get("cuda_peak")
            for section in (
                metrics.get("initialization", {}),
                metrics.get("generation", {}),
            )
            if section.get("cuda_peak") is not None
        ]
        metrics["cuda_peak_overall"] = {
            "allocated_gib": max(
                (record["allocated_gib"] for record in peak_records), default=0.0
            ),
            "reserved_gib": max(
                (record["reserved_gib"] for record in peak_records), default=0.0
            ),
        }
        metrics["budget_checks"] = {
            "gpu_reserved_within_budget": (
                metrics["cuda_peak_overall"]["reserved_gib"] <= args.gpu_budget_gib
            ),
            "cpu_rss_within_budget": (
                metrics["cpu_peak_rss_gib"] is not None
                and metrics["cpu_peak_rss_gib"] <= args.cpu_budget_gib
            ),
        }
        metrics["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_metrics(args.metrics_path, metrics)
        logger.info(
            "[METRICS] status={}, generation_latency={} s, "
            "peak_allocated={:.3f} GiB, peak_reserved={:.3f} GiB, "
            "cpu_peak_rss={} GiB, metrics={}",
            metrics["status"],
            metrics.get("generation", {}).get("elapsed_seconds"),
            metrics["cuda_peak_overall"]["allocated_gib"],
            metrics["cuda_peak_overall"]["reserved_gib"],
            metrics["cpu_peak_rss_gib"],
            args.metrics_path,
        )

    failed_budgets = [
        name for name, passed in metrics["budget_checks"].items() if not passed
    ]
    if failed_budgets:
        metrics["status"] = "budget_failed"
        metrics["budget_failure"] = failed_budgets
        write_metrics(args.metrics_path, metrics)
        logger.error(
            "[METRICS] memory budget validation failed: {}",
            ", ".join(failed_budgets),
        )
        raise RuntimeError("Memory budget validation failed: " + ", ".join(failed_budgets))


if __name__ == "__main__":
    main()
