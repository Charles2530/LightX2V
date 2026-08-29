"""Evaluate an exported FastWAM action policy with real LIBERO rollouts."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Each evaluator is intended to see one GPU. EGL then addresses that process-local
# device as zero even when CUDA_VISIBLE_DEVICES contains a nonzero physical GPU.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_SRC = ROOT / "lightx2v_ros" / "src" / "simulator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SIMULATOR_SRC) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SRC))

from simulator.libero_node.observer import (
    LIBERO_BENCHMARKS,
    LiberoActionObserver,
    default_libero_root,
    load_init_state,
)

from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
from lightx2v.utils.set_config import auto_calc_config, get_default_config

DEFAULT_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="FastWAM deployment JSON")
    parser.add_argument("--adapter", required=True, help="Exported native student checkpoint")
    parser.add_argument("--model-path", required=True, help="Wan2.2-TI2V-5B directory")
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--output", required=True, help="Rollout result JSON")
    parser.add_argument("--libero-root", default=str(default_libero_root()))
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Override task rollout horizon, excluding wait steps")
    parser.add_argument("--render-size", type=int, default=256, help="LIBERO camera render size before policy resize")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def atomic_json_dump(payload, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def quat_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator).astype(np.float32)


def policy_observation(obs):
    images = {
        "agentview": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
    }
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected 8-D LIBERO state, got {state.shape}")
    return images, state


def build_policy(args):
    config = get_default_config()
    config.update(
        {
            "model_cls": "fastwam",
            "task": "i2va",
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "config_json": str(Path(args.config).expanduser().resolve()),
        }
    )
    config = auto_calc_config(config)
    config.update(
        {
            "adapter_model_path": str(Path(args.adapter).expanduser().resolve()),
            "dataset_stats_path": str(Path(args.dataset_stats).expanduser().resolve()),
            "device": args.device,
            "seed": args.seed,
        }
    )
    return FastWAMPolicy.from_config(config), config


def summarize(episodes):
    groups = {}
    for episode in episodes:
        groups.setdefault(episode["benchmark"], []).append(episode)

    def score(items):
        successes = sum(bool(item["success"]) for item in items)
        return {"episodes": len(items), "successes": successes, "success_rate": successes / len(items) if items else 0.0}

    return {"overall": score(episodes), "benchmarks": {name: score(items) for name, items in sorted(groups.items())}}


def run_episode(observer, policy, *, wait_steps, max_steps):
    policy.reset()
    obs = observer.reset()
    dummy_action = np.zeros(7, dtype=np.float32)
    dummy_action[-1] = -1.0
    started = time.monotonic()
    for step in range(wait_steps + max_steps):
        if step < wait_steps:
            action = dummy_action
        else:
            images, state = policy_observation(obs)
            action = policy.next_action(images, state, observer.task_description)
        obs, _, success, _ = observer.step(action)
        if success:
            return True, step + 1, time.monotonic() - started
    return False, wait_steps + max_steps, time.monotonic() - started


def main():
    args = parse_args()
    unknown = sorted(set(args.benchmarks) - set(LIBERO_BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown LIBERO benchmarks: {unknown}")
    if args.episodes_per_task <= 0 or args.episode_offset < 0:
        raise ValueError("Episode count must be positive and offset must be non-negative")

    policy, policy_config = build_policy(args)
    wait_steps = int(policy_config.get("num_steps_wait", 30))
    results = {
        "adapter": str(Path(args.adapter).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "episodes_per_task": args.episodes_per_task,
        "episode_offset": args.episode_offset,
        "wait_steps": wait_steps,
        "render_size": args.render_size,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episodes": [],
        "summary": summarize([]),
    }
    atomic_json_dump(results, args.output)

    try:
        for benchmark in args.benchmarks:
            probe = LiberoActionObserver(
                benchmark_name=benchmark,
                task_id=0,
                init_state_id=0,
                image_size=args.render_size,
                seed=args.seed,
                libero_root=args.libero_root,
            )
            from libero.libero import get_libero_path

            num_tasks = probe.benchmark_module.get_benchmark_dict()[benchmark]().get_num_tasks()
            probe.close()
            task_ids = args.task_ids if args.task_ids is not None else range(num_tasks)
            for task_id in task_ids:
                if task_id < 0 or task_id >= num_tasks:
                    raise ValueError(f"task_id {task_id} is invalid for {benchmark} with {num_tasks} tasks")
                observer = LiberoActionObserver(
                    benchmark_name=benchmark,
                    task_id=task_id,
                    init_state_id=0,
                    image_size=args.render_size,
                    seed=args.seed,
                    libero_root=args.libero_root,
                )
                try:
                    _, num_init_states = load_init_state(get_libero_path, observer.task, 0)
                    horizon = args.max_steps or TASK_MAX_STEPS[benchmark]
                    for episode_index in range(args.episode_offset, args.episode_offset + args.episodes_per_task):
                        init_state_id = episode_index % num_init_states
                        init_state, _ = load_init_state(get_libero_path, observer.task, init_state_id)
                        observer.init_state_id = init_state_id
                        observer.init_state = np.asarray(init_state).copy()
                        success, steps, elapsed = run_episode(observer, policy, wait_steps=wait_steps, max_steps=horizon)
                        record = {
                            "benchmark": benchmark,
                            "task_id": task_id,
                            "task_name": observer.task.name,
                            "instruction": observer.task_description,
                            "episode_index": episode_index,
                            "init_state_id": init_state_id,
                            "success": success,
                            "steps": steps,
                            "elapsed_seconds": round(elapsed, 3),
                        }
                        results["episodes"].append(record)
                        results["summary"] = summarize(results["episodes"])
                        atomic_json_dump(results, args.output)
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                finally:
                    observer.close()
    finally:
        policy.close()

    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results["summary"] = summarize(results["episodes"])
    atomic_json_dump(results, args.output)
    print(json.dumps(results["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
