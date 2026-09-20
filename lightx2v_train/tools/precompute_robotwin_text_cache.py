"""Precompute RobotWin task text embeddings in parallel across local GPUs."""

import argparse
import os
from pathlib import Path

import torch

from lightx2v_train.data.libero.preparation import (
    _collect_prompts, _text_cache_path, _valid_text_cache, precompute_text_embeddings,
)
from lightx2v_train.data.robotwin_preparation import _dedupe_dataset_dirs
from lightx2v_train.runtime import load_config


def resolve_cache_dir(config, *, validate_only=False, environment=None):
    environment = os.environ if environment is None else environment
    configured = config["data"]["train"].get("text_embedding_cache_dir")
    fasterwam = config["model"].get("name") == "wan_fasterwam"
    # Preserve the original FastWAM tool's default; only FasterWAM opts into its separate configured path.
    default = (configured if fasterwam else None) or str(Path(config["training"]["output_dir"]) / "text_embeds_cache")
    selected = Path(environment.get("FASTWAM_TEXT_CACHE_DIR", default)).expanduser().resolve()
    if fasterwam and configured and not validate_only:
        if selected != Path(configured).expanduser().resolve():
            raise ValueError(
                "FasterWAM cache generation must use its configured text_embedding_cache_dir; "
                "unset the stale FASTWAM_TEXT_CACHE_DIR override to avoid writing the original FastWAM cache."
            )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true",
                        help="Read-only validation: fail on missing/invalid entries, never create or replace cache files.")
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
    cache_dir = resolve_cache_dir(config, validate_only=args.validate_only)
    print(f"rank={rank}/{world_size} prompts={len(prompts)} cache={cache_dir}", flush=True)
    model_path = os.environ.get("FASTWAM_TEXT_MODEL_PATH", config["model"]["model_path"])
    if args.validate_only:
        invalid = [prompt for prompt in prompts if not _valid_text_cache(_text_cache_path(cache_dir, prompt, 128), 128)]
        if invalid:
            raise FileNotFoundError(f"Read-only cache validation: invalid={len(invalid)}/{len(prompts)} cache={cache_dir}; no files changed.")
        print(f"rank={rank} read-only cache validation passed: {len(prompts)} entries; no files changed", flush=True)
    else:
        precompute_text_embeddings(model_path, cache_dir, 128, prompts)
    print(f"rank={rank} done", flush=True)


if __name__ == "__main__":
    main()
