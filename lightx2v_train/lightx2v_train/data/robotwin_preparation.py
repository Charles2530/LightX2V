import os
import shutil
from collections import defaultdict
from pathlib import Path

from loguru import logger

from lightx2v_train.data.libero.preparation import (
    _atomic_json_dump,
    _collect_prompts,
    _is_missing,
    _resolve_path,
    _text_cache_path,
    _validate_text_caches,
    calculate_dataset_stats,
    precompute_text_embeddings,
)
from lightx2v_train.runtime.distributed import barrier, is_main_process


def _split_configs(config):
    data_config = config["data"]
    return [split_config for split_config in (data_config.get("train"), data_config.get("val")) if split_config is not None]


def _cache_groups(split_configs):
    grouped = defaultdict(list)
    for split_config in split_configs:
        cache_dir = _resolve_path(split_config["text_embedding_cache_dir"])
        context_len = int(split_config.get("context_len", 128))
        grouped[(cache_dir, context_len)].append(split_config)
    return grouped.items()


def _dedupe_dataset_dirs(split_configs):
    deduped = []
    seen = set()
    for split_config in split_configs:
        dataset_dirs = split_config.get("dataset_dirs")
        if isinstance(dataset_dirs, (str, Path)):
            dataset_dirs = [dataset_dirs]
        clone = dict(split_config)
        clone_dirs = []
        for dataset_dir in dataset_dirs or []:
            resolved = str(_resolve_path(dataset_dir))
            if resolved in seen:
                continue
            seen.add(resolved)
            clone_dirs.append(resolved)
        if clone_dirs:
            clone["dataset_dirs"] = clone_dirs
            deduped.append(clone)
    return deduped


def _validate_text_cache_presence(split_configs):
    for (cache_dir, context_len), grouped_splits in _cache_groups(split_configs):
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"RobotWin text embedding cache directory does not exist: {cache_dir}")
        prompts = _collect_prompts(_dedupe_dataset_dirs(grouped_splits))
        existing_names = set(os.listdir(cache_dir))
        first_missing = None
        missing_count = 0
        for prompt in prompts:
            path = _text_cache_path(cache_dir, prompt, context_len)
            if path.name not in existing_names:
                missing_count += 1
                if first_missing is None:
                    first_missing = path
        if missing_count:
            raise FileNotFoundError(
                "RobotWin text embedding cache is incomplete: "
                f"missing={missing_count}/{len(prompts)} first={first_missing}"
            )
        logger.info(
            "[data-preflight] RobotWin text embedding cache ready: prompts={} existing_entries={} context_len={} path={}",
            len(prompts),
            len(existing_names),
            context_len,
            cache_dir,
        )


def _copy_or_calculate_stats(config, train_config, val_config, output_dir):
    configured_stats = train_config.get("pretrained_norm_stats")
    if _is_missing(configured_stats):
        stats_path = output_dir / "dataset_stats.json"
        if is_main_process() and not stats_path.is_file():
            logger.info("[data-preflight] calculating RobotWin normalization statistics")
            stats = calculate_dataset_stats(train_config)
            _atomic_json_dump(stats, stats_path)
            logger.info(
                "[data-preflight] saved normalization statistics: episodes={} transitions={} path={}",
                stats["num_episodes"],
                stats["num_transition"],
                stats_path,
            )
        barrier()
    else:
        stats_path = _resolve_path(configured_stats)
        if not stats_path.is_file():
            raise FileNotFoundError(f"Configured RobotWin normalization stats do not exist: {stats_path}")
        eval_stats_path = output_dir / "dataset_stats.json"
        if is_main_process() and stats_path != eval_stats_path:
            eval_stats_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stats_path, eval_stats_path)
        barrier()

    train_config["pretrained_norm_stats"] = str(stats_path)
    if val_config is not None and _is_missing(val_config.get("pretrained_norm_stats")):
        val_config["pretrained_norm_stats"] = str(stats_path)


def _prepare_text_cache(config, train_config, val_config, output_dir):
    configured_cache = train_config.get("text_embedding_cache_dir")
    auto_text_cache = _is_missing(configured_cache)
    cache_dir = output_dir / "text_embeds_cache" if auto_text_cache else _resolve_path(configured_cache)
    train_config["text_embedding_cache_dir"] = str(cache_dir)
    if val_config is not None and _is_missing(val_config.get("text_embedding_cache_dir")):
        val_config["text_embedding_cache_dir"] = str(cache_dir)

    split_configs = _split_configs(config)
    if auto_text_cache:
        for (group_cache_dir, context_len), grouped_splits in _cache_groups(split_configs):
            prompts = _collect_prompts(_dedupe_dataset_dirs(grouped_splits))
            if is_main_process():
                precompute_text_embeddings(
                    model_path=config["model"]["model_path"],
                    cache_dir=group_cache_dir,
                    context_len=context_len,
                    prompts=prompts,
                )
            barrier()

    if is_main_process():
        _validate_text_cache_presence(split_configs)
        if any(bool(split_config.get("validate_text_cache_shapes", False)) for split_config in split_configs):
            logger.info("[data-preflight] RobotWin running full text embedding shape validation")
            _validate_text_caches(split_configs)
    barrier()


def prepare_robotwin_fastwam_assets(config):
    train_config = config["data"]["train"]
    val_config = config["data"].get("val")
    output_dir = _resolve_path(config["training"]["output_dir"])

    _copy_or_calculate_stats(config, train_config, val_config, output_dir)
    _prepare_text_cache(config, train_config, val_config, output_dir)
