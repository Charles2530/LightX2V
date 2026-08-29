"""Evaluate an exported FastWAM action policy with real LIBERO rollouts."""

import argparse
import json
import math
import os
import random
import shlex
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
SIMULATOR_SRC = ROOT / "lightx2v_ros" / "src" / "simulator"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SIMULATOR_SRC) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SRC))

from simulator.libero_node.observer import (
    LIBERO_BENCHMARKS,
    LiberoActionObserver,
    create_task_suite,
    default_libero_root,
    load_libero,
)

from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
from lightx2v.utils.set_config import auto_calc_config, get_default_config
from libero_eval_protocol import (
    load_fastwam_evaluation_implementation,
    load_official_evaluation_protocol,
)

DEFAULT_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


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
    parser.add_argument("--max-steps", type=int, default=0, help="Override the official policy-step horizon")
    parser.add_argument("--render-size", type=int, default=256, help="LIBERO camera render size before policy resize")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-action-infer-steps", type=int, default=1)
    parser.add_argument("--release-text-encoder-after-prompt-cache", action="store_true")
    parser.add_argument("--ready-file", help="Write a readiness marker after model and prompt initialization")
    return parser.parse_args()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolved(path):
    return str(Path(path).expanduser().resolve())


def atomic_json_dump(payload, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_task_metadata(libero_root):
    path = Path(libero_root).expanduser().resolve() / "libero" / "libero" / "benchmark" / "task_classification.json"
    if not path.is_file():
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = {}
    for benchmark, items in payload.items():
        for task_id, item in enumerate(items):
            official_id = int(item["id"])
            if official_id != task_id + 1:
                raise ValueError(f"Non-contiguous task id in {path}: {benchmark} id={official_id}")
            metadata[(benchmark, task_id)] = {
                "classification_id": official_id,
                "category": item.get("category"),
                "difficulty_level": item.get("difficulty_level"),
                "classified_task_name": item["name"],
            }
    return metadata, str(path)


def discover_task_counts(libero_root, benchmarks):
    benchmark_module, _, _ = load_libero(Path(libero_root).expanduser().resolve())
    factories = benchmark_module.get_benchmark_dict()
    counts = {}
    for benchmark in benchmarks:
        if benchmark not in factories:
            raise ValueError(f"Benchmark {benchmark!r} is not registered by {libero_root}")
        counts[benchmark] = create_task_suite(factories[benchmark]).get_num_tasks()
    return counts


def load_task_descriptions(libero_root, task_ids_by_benchmark):
    benchmark_module, _, _ = load_libero(Path(libero_root).expanduser().resolve())
    factories = benchmark_module.get_benchmark_dict()
    descriptions = []
    for benchmark, task_ids in task_ids_by_benchmark.items():
        task_suite = create_task_suite(factories[benchmark])
        descriptions.extend(task_suite.get_task(task_id).language for task_id in task_ids)
    return descriptions


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
            "model_path": resolved(args.model_path),
            "config_json": resolved(args.config),
        }
    )
    config = auto_calc_config(config)
    config.update(
        {
            "adapter_model_path": resolved(args.adapter),
            "dataset_stats_path": resolved(args.dataset_stats),
            "device": args.device,
            "seed": args.seed,
        }
    )
    return FastWAMPolicy.from_config(config), config


def score(items):
    successes = sum(bool(item["success"]) for item in items)
    elapsed = [float(item["elapsed_seconds"]) for item in items]
    steps = [int(item["steps"]) for item in items]
    success_steps = [int(item["steps"]) for item in items if item["success"]]
    failures = Counter(item.get("failure_reason") or "unknown" for item in items if not item["success"])
    return {
        "episodes": len(items),
        "successes": successes,
        "failures": len(items) - successes,
        "success_rate": successes / len(items) if items else 0.0,
        "average_steps": sum(steps) / len(steps) if steps else 0.0,
        "average_success_steps": sum(success_steps) / len(success_steps) if success_steps else None,
        "elapsed_seconds": round(sum(elapsed), 3),
        "average_elapsed_seconds": round(sum(elapsed) / len(elapsed), 3) if elapsed else 0.0,
        "failure_reasons": dict(sorted(failures.items())),
    }


def grouped_scores(episodes, key):
    groups = {}
    for episode in episodes:
        value = key(episode)
        if value is not None:
            groups.setdefault(str(value), []).append(episode)
    return {name: score(items) for name, items in sorted(groups.items())}


def summarize(episodes):
    tasks = grouped_scores(episodes, lambda item: f"{item['benchmark']}/{item['task_id']}")
    for task_key, task_score in tasks.items():
        first = next(item for item in episodes if f"{item['benchmark']}/{item['task_id']}" == task_key)
        task_score.update(
            {
                "task_name": first["task_name"],
                "category": first.get("category"),
                "difficulty_level": first.get("difficulty_level"),
            }
        )
    return {
        "overall": score(episodes),
        "benchmarks": grouped_scores(episodes, lambda item: item["benchmark"]),
        "tasks": tasks,
        "categories": grouped_scores(episodes, lambda item: item.get("category")),
        "difficulty_levels": grouped_scores(
            episodes,
            lambda item: "unlabeled" if item.get("difficulty_level") is None else item["difficulty_level"],
        ),
    }


def run_episode(observer, policy, *, initialization_steps, max_steps):
    policy.reset()
    obs = observer.reset()
    dummy_action = np.zeros(7, dtype=np.float32)
    started = time.monotonic()
    # Match libero/lifelong/metric.py: settle physics with five all-zero actions.
    for _ in range(initialization_steps):
        obs, _, _, _ = observer.step(dummy_action)
    for policy_step in range(max_steps):
        images, state = policy_observation(obs)
        action = policy.next_action(images, state, observer.task_description)
        obs, _, success, _ = observer.step(action)
        if success:
            return True, policy_step + 1, time.monotonic() - started
    return False, max_steps, time.monotonic() - started


def build_run_signature(args, task_ids_by_benchmark, official_protocol, implementation):
    return {
        "adapter": resolved(args.adapter),
        "config": resolved(args.config),
        "model_path": resolved(args.model_path),
        "dataset_stats": resolved(args.dataset_stats),
        "libero_root": resolved(args.libero_root),
        "benchmarks": list(args.benchmarks),
        "task_ids": task_ids_by_benchmark,
        "episodes_per_task": args.episodes_per_task,
        "episode_offset": args.episode_offset,
        "max_steps_override": args.max_steps,
        "render_size": args.render_size,
        "seed": args.seed,
        "expected_action_infer_steps": args.expected_action_infer_steps,
        "release_text_encoder_after_prompt_cache": args.release_text_encoder_after_prompt_cache,
        "official_evaluation": official_protocol,
        "fastwam_evaluation_implementation": implementation,
    }


def load_or_create_results(output, run_signature, protocol):
    output = Path(output).expanduser().resolve()
    if output.is_file():
        results = json.loads(output.read_text(encoding="utf-8"))
        if results.get("run_signature") != run_signature:
            raise RuntimeError(f"Existing output has a different run signature: {output}")
        if results.get("protocol", {}).get("action_infer_steps") != protocol["action_infer_steps"]:
            raise RuntimeError(f"Existing output has a different action_infer_steps: {output}")
        results["protocol"] = protocol
        results.pop("finished_at", None)
        return results
    return {
        "run_signature": run_signature,
        "protocol": protocol,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "episodes": [],
        "summary": summarize([]),
    }


def main():
    args = parse_args()
    if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError(
                "Set MUJOCO_EGL_DEVICE_ID explicitly when CUDA_VISIBLE_DEVICES is set; "
                "LIBERO-plus evaluation normally uses unfiltered physical device ids."
            )
        if not args.device.startswith("cuda:"):
            raise ValueError("--device must be an explicit cuda:N device for EGL evaluation")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = args.device.split(":", 1)[1]
    unknown = sorted(set(args.benchmarks) - set(LIBERO_BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown LIBERO benchmarks: {unknown}")
    if args.episodes_per_task <= 0 or args.episode_offset < 0:
        raise ValueError("Episode count must be positive and offset must be non-negative")
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")

    official_protocol = load_official_evaluation_protocol(args.libero_root)
    implementation = load_fastwam_evaluation_implementation(ROOT)
    task_counts = discover_task_counts(args.libero_root, args.benchmarks)
    task_ids_by_benchmark = {}
    for benchmark in args.benchmarks:
        task_ids = list(args.task_ids) if args.task_ids is not None else list(range(task_counts[benchmark]))
        invalid = [task_id for task_id in task_ids if task_id < 0 or task_id >= task_counts[benchmark]]
        if invalid:
            raise ValueError(f"Invalid task ids for {benchmark} ({task_counts[benchmark]} tasks): {invalid}")
        task_ids_by_benchmark[benchmark] = task_ids

    random.seed(args.seed)
    np.random.seed(args.seed)
    metadata, classification_path = load_task_metadata(args.libero_root)
    policy, policy_config = build_policy(args)
    prompt_cache_size = policy.preload_task_prompts(
        load_task_descriptions(args.libero_root, task_ids_by_benchmark),
        release_text_encoder=args.release_text_encoder_after_prompt_cache,
    )
    configured_wait_steps = int(policy_config.get("num_steps_wait", 0))
    action_infer_steps = int(policy.action_infer_steps)
    if action_infer_steps != args.expected_action_infer_steps:
        policy.close()
        raise RuntimeError(
            f"Expected action_infer_steps={args.expected_action_infer_steps}, got {action_infer_steps} from {resolved(args.config)}"
        )

    protocol = {
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "adapter": resolved(args.adapter),
        "config": resolved(args.config),
        "model_path": resolved(args.model_path),
        "dataset_stats": resolved(args.dataset_stats),
        "libero_root": resolved(args.libero_root),
        "task_classification": classification_path,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_cuda_device": (os.environ.get("CUDA_VISIBLE_DEVICES") or args.device.split(":", 1)[1]).split(",")[0],
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "seed": args.seed,
        "action_infer_steps": action_infer_steps,
        "actions_per_plan": int(policy.actions_per_plan),
        "action_chunk_size": int(policy.action_chunk_size),
        "configured_num_steps_wait_ignored": configured_wait_steps,
        "prompt_cache_size": prompt_cache_size,
        "text_encoder_released": policy.text_encoder_released,
        "official_evaluation": official_protocol,
        "fastwam_evaluation_implementation": implementation,
        "max_policy_steps": args.max_steps or official_protocol["max_policy_steps"],
        "render_size": args.render_size,
        "task_counts": task_counts,
    }
    print(f"[protocol] {json.dumps(protocol, ensure_ascii=False, sort_keys=True)}", flush=True)

    run_signature = build_run_signature(args, task_ids_by_benchmark, official_protocol, implementation)
    results = load_or_create_results(args.output, run_signature, protocol)
    completed = {
        (item["benchmark"], int(item["task_id"]), int(item["episode_index"])) for item in results["episodes"]
    }
    expected_episode_ids = range(args.episode_offset, args.episode_offset + args.episodes_per_task)
    atomic_json_dump(results, args.output)
    if args.ready_file:
        atomic_json_dump(
            {
                "ready_at": utc_now(),
                "pid": os.getpid(),
                "output": resolved(args.output),
                "prompt_cache_size": prompt_cache_size,
                "text_encoder_released": policy.text_encoder_released,
                "action_infer_steps": action_infer_steps,
            },
            args.ready_file,
        )

    try:
        for benchmark in args.benchmarks:
            horizon = args.max_steps or official_protocol["max_policy_steps"]
            for task_id in task_ids_by_benchmark[benchmark]:
                pending_episode_ids = [
                    episode_index
                    for episode_index in expected_episode_ids
                    if (benchmark, task_id, episode_index) not in completed
                ]
                if not pending_episode_ids:
                    continue
                observer = LiberoActionObserver(
                    benchmark_name=benchmark,
                    task_id=task_id,
                    init_state_id=0,
                    image_size=args.render_size,
                    seed=args.seed,
                    libero_root=args.libero_root,
                )
                try:
                    task_metadata = metadata.get((benchmark, task_id), {})
                    classified_name = task_metadata.get("classified_task_name")
                    if classified_name is not None and classified_name != observer.task.name:
                        raise RuntimeError(
                            f"Task classification mismatch for {benchmark}/{task_id}: "
                            f"{classified_name!r} != {observer.task.name!r}"
                        )
                    for episode_index in pending_episode_ids:
                        init_state_id = episode_index % observer.num_init_states
                        observer.set_init_state_id(init_state_id)
                        success, steps, elapsed = run_episode(
                            observer,
                            policy,
                            initialization_steps=official_protocol["initialization_steps"],
                            max_steps=horizon,
                        )
                        record = {
                            "benchmark": benchmark,
                            "task_id": task_id,
                            "classification_id": task_metadata.get("classification_id"),
                            "task_name": observer.task.name,
                            "instruction": observer.task_description,
                            "category": task_metadata.get("category"),
                            "difficulty_level": task_metadata.get("difficulty_level"),
                            "episode_index": episode_index,
                            "init_state_id": init_state_id,
                            "num_init_states": observer.num_init_states,
                            "seed": args.seed,
                            "success": bool(success),
                            "failure_reason": None if success else "max_steps_exceeded",
                            "steps": steps,
                            "policy_steps": steps,
                            "initialization_steps": official_protocol["initialization_steps"],
                            "total_env_steps": official_protocol["initialization_steps"] + steps,
                            "max_policy_steps": horizon,
                            "elapsed_seconds": round(elapsed, 3),
                        }
                        results["episodes"].append(record)
                        completed.add((benchmark, task_id, episode_index))
                        results["updated_at"] = utc_now()
                        results["summary"] = summarize(results["episodes"])
                        atomic_json_dump(results, args.output)
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                finally:
                    observer.close()
    finally:
        policy.close()

    results["finished_at"] = utc_now()
    results["updated_at"] = results["finished_at"]
    results["summary"] = summarize(results["episodes"])
    atomic_json_dump(results, args.output)
    print(json.dumps(results["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
