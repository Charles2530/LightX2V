"""Replay one validation prompt from DMD student checkpoints on a new GPU count."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from lightx2v_train.data import build_data, build_sample_processor
from lightx2v_train.infer import build_inferencer
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config, setup_logger
from lightx2v_train.runtime.fsdp import apply_fsdp2
from lightx2v_train.runtime.precision import configure_precision
from lightx2v_train.utils.quantization import replace_blocks_linear, validate_training_quantization


def select_prompt(source: Path, index: int, target: Path) -> str:
    with source.open(encoding="utf-8") as stream:
        for row, line in enumerate(stream):
            if row == index:
                sample = json.loads(line)
                prompt = sample["prompt"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
                return prompt
    raise IndexError(f"prompt index {index} is outside {source}")


def load_student(model, checkpoint: Path, process_group):
    metadata = FileSystemReader(str(checkpoint)).read_metadata()
    stored = {key.removeprefix("student_model.") for key in metadata.state_dict_metadata if key.startswith("student_model.")}
    module = model.fsdp2_state_module()
    options = StateDictOptions(ignore_frozen_params=True, strict=False)
    state = get_model_state_dict(module, options=options)
    if set(state) != stored:
        raise RuntimeError(f"student checkpoint key mismatch: model={len(state)} checkpoint={len(stored)}")
    dcp.load({"student_model": state}, checkpoint_id=str(checkpoint), process_group=process_group)
    set_model_state_dict(module, state, options=options)
    logger.info("Loaded student weights from {} ({} tensors)", checkpoint, len(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, default=33)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", default=(0, 100, 500, 1000))
    args = parser.parse_args()
    checkpoint_100 = args.run_dir / "checkpoint-000000100"
    recorded = torch.load(checkpoint_100 / "trainer_state.pt", map_location="cpu", weights_only=True)
    config = load_config(str(checkpoint_100 / "config.yaml"))
    config["quant"] = recorded["quant"]
    config["distributed"]["fsdp2"]["size"] = int(os.environ.get("WORLD_SIZE", "1"))
    config["data"]["val"]["data_path"] = [str(args.output_dir / "selected_prompt.jsonl")]
    config["data"]["val"]["max_samples"] = 1
    config["inference"]["output_dir"] = str(args.output_dir)
    config["inference"]["seed"] = 42
    config["training"]["output_dir"] = str(args.output_dir / "logs")
    configure_precision(config, section="inference")
    init_distributed(config)
    setup_logger(config)
    try:
        if dist.get_rank() == 0:
            prompt = select_prompt(args.prompt_file, args.prompt_index, args.output_dir / "selected_prompt.jsonl")
            logger.info("Selected validation prompt index={} text={}", args.prompt_index, prompt[:160])
        dist.barrier()
        quant = validate_training_quantization(config)
        model = build_model(config)
        model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=True)
        if quant is not None:
            count = replace_blocks_linear(model.denoiser_module(), quant)
            logger.info("Enabled student QAT inference on {} Linear layers", count)
        apply_fsdp2(model, config)
        dataloader = build_data(config, train_or_val="val", sample_processor=build_sample_processor(config))
        checkpoint_group = dist.new_group(backend="gloo")
        for iteration in args.iterations:
            output = args.output_dir / f"iter-{iteration:09d}"
            video = output / "00000.mp4"
            if video.is_file() and video.stat().st_size:
                logger.info("Existing video: {}", video)
                continue
            if iteration:
                checkpoint = args.run_dir / f"checkpoint-{iteration:09d}"
                if not checkpoint.is_dir():
                    logger.info("Checkpoint unavailable at iteration {}", iteration)
                    continue
                state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True)
                if state["iteration"] != iteration or state["quant"] != quant:
                    raise ValueError(f"checkpoint configuration mismatch: {checkpoint}")
                load_student(model, checkpoint / "dist_state", checkpoint_group)
            config["inference"]["output_dir"] = str(output)
            inferencer = build_inferencer(config)
            inferencer.set_model(model)
            inferencer.set_data(dataloader)
            inferencer.infer()
            if dist.get_rank() == 0:
                logger.info("Generated iteration {}: {}", iteration, video)
        if dist.get_rank() == 0:
            (args.output_dir / "provenance.json").write_text(json.dumps({
                "source_run": str(args.run_dir.resolve()), "prompt_file": str(args.prompt_file.resolve()),
                "prompt_index_zero_based": args.prompt_index, "seed": 42, "iterations": args.iterations,
                "quant": quant, "checkpoint_loading": "PyTorch DCP student_model only, resharded to WORLD_SIZE GPUs",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
