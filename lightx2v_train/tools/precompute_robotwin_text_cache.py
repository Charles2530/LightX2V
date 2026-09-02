"""Precompute RobotWin task text embeddings in parallel across local GPUs."""

import argparse
import os
from pathlib import Path

import torch

from lightx2v_train.data.libero.preparation import _collect_prompts, precompute_text_embeddings
from lightx2v_train.data.robotwin_preparation import _dedupe_dataset_dirs
from lightx2v_train.runtime import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for text embedding precomputation")

    # torchrun supplies LOCAL_RANK; each process owns one GPU and a disjoint
    # prompt partition.  No process needs to initialize a process group.
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    splits = [config["data"][key] for key in ("train", "val") if config["data"].get(key) is not None]
    prompts = _collect_prompts(_dedupe_dataset_dirs(splits))
    prompts = prompts[rank::world_size]
    cache_dir = Path(
        os.environ.get(
            "FASTWAM_TEXT_CACHE_DIR",
            str(Path(config["training"]["output_dir"]).expanduser().resolve() / "text_embeds_cache"),
        )
    ).expanduser().resolve()
    print(f"rank={rank}/{world_size} prompts={len(prompts)} cache={cache_dir}", flush=True)
    model_path = os.environ.get("FASTWAM_TEXT_MODEL_PATH", config["model"]["model_path"])
    precompute_text_embeddings(model_path, cache_dir, 128, prompts)
    print(f"rank={rank} done", flush=True)


if __name__ == "__main__":
    main()
