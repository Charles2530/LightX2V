import argparse
import time

import torch
from lightx2v_train.data import build_data, prepare_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import (
    CachedActionDenoiser,
    build_action_distill_condition,
    sample_action_teacher,
)
from lightx2v_train.runtime import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FastWAM action sampling at several step counts.")
    parser.add_argument("--config", required=True, help="FastWAM training config with model and validation data paths.")
    parser.add_argument("--checkpoint", help="Optional FastWAM checkpoint overriding model.checkpoint_path.")
    parser.add_argument("--steps", default="1,2,4,5,10,20")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _masked_l1_per_sample(left, right, valid_mask):
    error = (left.float() - right.float()).abs()
    reduce_dims = tuple(range(1, error.ndim))
    if valid_mask is None:
        return error.mean(dim=reduce_dims)
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1).expand_as(error)
    return (error * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp(min=1)


def _slice_batch(value, size):
    if isinstance(value, torch.Tensor):
        return value[:size]
    if isinstance(value, dict):
        return {key: _slice_batch(item, size) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return value[:size]
    return value


def main():
    args = parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    if not steps or min(steps) <= 0:
        raise ValueError("--steps must contain positive integers.")
    config = load_config(args.config)
    if args.checkpoint:
        config["model"]["checkpoint_path"] = args.checkpoint
    prepare_data(config)
    model = build_model(config)
    model.load_components()
    dataloader = build_data(config, train_or_val="val")
    module = model.unwrap_module().eval()
    denoiser = CachedActionDenoiser(module.action_expert, module.mot).eval()
    generator = torch.Generator(device=module.device).manual_seed(args.seed)
    totals = {step: {"gt_l1": 0.0, "latency_ms": 0.0} for step in steps}
    count = 0

    with torch.no_grad(), model.autocast_context():
        for sample in dataloader:
            batch_size = int(sample["video"].shape[0])
            remaining = args.num_samples - count
            if batch_size > remaining:
                sample = _slice_batch(sample, remaining)
                batch_size = remaining
            inputs = module.build_action_distill_inputs(sample)
            condition = build_action_distill_condition(module, inputs)
            valid_mask = None if inputs["action_is_pad"] is None else ~inputs["action_is_pad"]
            noise = torch.randn(inputs["action"].shape, generator=generator, device=module.device, dtype=module.torch_dtype)
            for step in steps:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started_at = time.perf_counter()
                action = sample_action_teacher(denoiser, noise.clone(), condition, module.infer_action_scheduler, step)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                totals[step]["latency_ms"] += (time.perf_counter() - started_at) * 1000.0
                totals[step]["gt_l1"] += float(_masked_l1_per_sample(action, inputs["action"], valid_mask).sum().item())
            count += batch_size
            if count >= args.num_samples:
                break

    if count == 0:
        raise RuntimeError("Validation dataloader produced no samples.")
    for step in steps:
        print(f"steps={step:2d} samples={count} action_gt_l1={totals[step]['gt_l1'] / count:.6f} denoise_latency_ms_per_sample={totals[step]['latency_ms'] / count:.3f}")


if __name__ == "__main__":
    main()
