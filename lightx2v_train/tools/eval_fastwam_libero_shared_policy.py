"""Evaluate LIBERO tasks with one FastWAM policy shared by many env workers.

The policy remains in the parent process. Spawned workers own independent
LIBERO environments and request deterministic action chunks over queues. This
keeps the official episode protocol while avoiding one 5B model replica per
environment worker.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import shlex
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
SIMULATOR_SRC = ROOT / "lightx2v_ros" / "src" / "simulator"
for path in (TOOLS_ROOT, ROOT, SIMULATOR_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--tasks-per-shard", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-action-infer-steps", type=int, default=1)
    parser.add_argument("--expected-actions-per-plan", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env-workers", type=int, default=12)
    parser.add_argument("--prompt-cache-limit", type=int, default=256)
    parser.add_argument("--assignment-index", type=int, default=0)
    parser.add_argument("--assignment-count", type=int, default=1)
    parser.add_argument("--ready-file")
    return parser.parse_args()


def resolved(path):
    return str(Path(path).expanduser().resolve())


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json_dump(payload, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def evaluation_implementation():
    from libero_eval_protocol import load_fastwam_evaluation_implementation

    implementation = load_fastwam_evaluation_implementation(ROOT)
    relative = "lightx2v_train/tools/eval_fastwam_libero_shared_policy.py"
    implementation[relative] = file_record(ROOT / relative)
    return implementation


def discover_selected_tasks(args):
    from simulator.libero_node.observer import create_task_suite, load_libero

    benchmark_module, _, _ = load_libero(Path(args.libero_root).expanduser().resolve())
    factories = benchmark_module.get_benchmark_dict()
    selected = {}
    for benchmark in args.benchmarks:
        if benchmark not in factories:
            raise ValueError(f"Unknown LIBERO benchmark: {benchmark}")
        count = create_task_suite(factories[benchmark]).get_num_tasks()
        if args.task_ids is not None:
            if len(args.benchmarks) != 1:
                raise ValueError("--task-ids requires exactly one benchmark")
            invalid = [task_id for task_id in args.task_ids if task_id < 0 or task_id >= count]
            if invalid:
                raise ValueError(f"Invalid task ids for {benchmark}: {invalid}")
            selected[benchmark] = list(dict.fromkeys(args.task_ids))
        else:
            selected[benchmark] = list(range(count))
    return selected


def build_shards(selected_tasks, tasks_per_shard):
    shards = []
    for benchmark, task_ids in selected_tasks.items():
        for start in range(0, len(task_ids), tasks_per_shard):
            chunk = tuple(task_ids[start : start + tasks_per_shard])
            name = f"{benchmark}-tasks-{chunk[0]:05d}-{chunk[-1]:05d}"
            shards.append({"name": name, "benchmark": benchmark, "task_ids": chunk})
    return shards


def load_task_metadata(libero_root):
    path = Path(libero_root).expanduser().resolve() / "libero" / "libero" / "benchmark" / "task_classification.json"
    if not path.is_file():
        return {}
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
    return metadata


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


def score(items):
    successes = sum(bool(item["success"]) for item in items)
    steps = [int(item["steps"]) for item in items]
    success_steps = [int(item["steps"]) for item in items if item["success"]]
    elapsed = [float(item["elapsed_seconds"]) for item in items]
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
    examples = {}
    for episode in episodes:
        examples.setdefault(f"{episode['benchmark']}/{episode['task_id']}", episode)
    for task_key, task_score in tasks.items():
        first = examples[task_key]
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


def shard_run_signature(args, shard, official_protocol, implementation):
    return {
        "adapter": resolved(args.adapter),
        "config": resolved(args.config),
        "model_path": resolved(args.model_path),
        "dataset_stats": resolved(args.dataset_stats),
        "libero_root": resolved(args.libero_root),
        "benchmarks": [shard["benchmark"]],
        "task_ids": {shard["benchmark"]: list(shard["task_ids"])},
        "episodes_per_task": args.episodes_per_task,
        "episode_offset": args.episode_offset,
        "max_steps_override": args.max_steps,
        "render_size": args.render_size,
        "seed": args.seed,
        "expected_action_infer_steps": args.expected_action_infer_steps,
        "official_evaluation": official_protocol,
        "fastwam_evaluation_implementation": implementation,
        "shared_policy": {
            "transport": "multiprocessing-queues-v1",
            "actions_per_plan": None,
            "prompt_cache_limit": args.prompt_cache_limit,
        },
    }


def shard_is_complete(path, shard, args, official_protocol, implementation, actions_per_plan):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected_signature = shard_run_signature(args, shard, official_protocol, implementation)
    expected_signature["shared_policy"]["actions_per_plan"] = int(actions_per_plan)
    if not payload.get("finished_at") or payload.get("run_signature") != expected_signature:
        return False
    expected = {
        (shard["benchmark"], task_id, episode_index)
        for task_id in shard["task_ids"]
        for episode_index in range(args.episode_offset, args.episode_offset + args.episodes_per_task)
    }
    actual = {
        (item["benchmark"], int(item["task_id"]), int(item["episode_index"]))
        for item in payload.get("episodes", [])
    }
    protocol = payload.get("protocol", {})
    return (
        actual == expected
        and int(protocol.get("action_infer_steps", -1)) == args.expected_action_infer_steps
        and int(protocol.get("actions_per_plan", -1)) == int(actions_per_plan)
        and int(protocol.get("seed", -1)) == args.seed
        and protocol.get("official_evaluation") == official_protocol
        and protocol.get("fastwam_evaluation_implementation") == implementation
    )


def load_or_create_results(path, run_signature, protocol):
    path = Path(path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_signature") != run_signature:
            raise RuntimeError(f"Existing output has a different run signature: {path}")
        payload["protocol"] = protocol
        payload.pop("finished_at", None)
        return payload
    return {
        "run_signature": run_signature,
        "protocol": protocol,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "episodes": [],
        "summary": summarize([]),
    }


def request_action_chunk(request_queue, response_queue, worker_index, request_id, images, state, task_description):
    request_queue.put((worker_index, request_id, images, state, task_description))
    response_id, actions, error = response_queue.get()
    if response_id != request_id:
        raise RuntimeError(f"Response id mismatch: expected {request_id}, got {response_id}")
    if error is not None:
        raise RuntimeError(f"Shared policy request failed: {error}")
    return np.asarray(actions, dtype=np.float32)


def run_shared_episode(
    observer,
    request_queue,
    response_queue,
    worker_index,
    request_counter,
    *,
    actions_per_plan,
    initialization_steps,
    max_steps,
):
    pending_actions = deque()
    obs = observer.reset()
    dummy_action = np.zeros(7, dtype=np.float32)
    started = time.monotonic()
    for _ in range(initialization_steps):
        obs, _, _, _ = observer.step(dummy_action)
    for policy_step in range(max_steps):
        if not pending_actions:
            images, state = policy_observation(obs)
            request_id = next(request_counter)
            chunk = request_action_chunk(
                request_queue,
                response_queue,
                worker_index,
                request_id,
                images,
                state,
                observer.task_description,
            )
            for action in chunk[:actions_per_plan]:
                pending_actions.append(np.asarray(action, dtype=np.float32))
        action = pending_actions.popleft()
        obs, _, success, _ = observer.step(action)
        if success:
            return True, policy_step + 1, time.monotonic() - started
    return False, max_steps, time.monotonic() - started


def counter(start=0):
    value = int(start)
    while True:
        yield value
        value += 1


def evaluate_shard(job, args_payload, request_queue, response_queue, worker_index, metadata):
    from simulator.libero_node.observer import LiberoActionObserver

    shard = job["shard"]
    output = Path(job["output"])
    protocol = job["protocol"]
    run_signature = job["run_signature"]
    official = protocol["official_evaluation"]
    results = load_or_create_results(output, run_signature, protocol)
    completed = {
        (item["benchmark"], int(item["task_id"]), int(item["episode_index"]))
        for item in results["episodes"]
    }
    episode_ids = range(
        args_payload["episode_offset"],
        args_payload["episode_offset"] + args_payload["episodes_per_task"],
    )
    request_counter = counter()
    horizon = args_payload["max_steps"] or official["max_policy_steps"]

    atomic_json_dump(results, output)
    for task_id in shard["task_ids"]:
        pending_episode_ids = [
            episode_index
            for episode_index in episode_ids
            if (shard["benchmark"], task_id, episode_index) not in completed
        ]
        if not pending_episode_ids:
            continue
        observer = LiberoActionObserver(
            benchmark_name=shard["benchmark"],
            task_id=task_id,
            init_state_id=0,
            image_size=args_payload["render_size"],
            seed=args_payload["seed"],
            libero_root=args_payload["libero_root"],
        )
        try:
            task_metadata = metadata.get((shard["benchmark"], task_id), {})
            classified_name = task_metadata.get("classified_task_name")
            if classified_name is not None and classified_name != observer.task.name:
                raise RuntimeError(
                    f"Task classification mismatch for {shard['benchmark']}/{task_id}: "
                    f"{classified_name!r} != {observer.task.name!r}"
                )
            for episode_index in pending_episode_ids:
                init_state_id = episode_index % observer.num_init_states
                observer.set_init_state_id(init_state_id)
                success, steps, elapsed = run_shared_episode(
                    observer,
                    request_queue,
                    response_queue,
                    worker_index,
                    request_counter,
                    actions_per_plan=protocol["actions_per_plan"],
                    initialization_steps=official["initialization_steps"],
                    max_steps=horizon,
                )
                record = {
                    "benchmark": shard["benchmark"],
                    "task_id": task_id,
                    "classification_id": task_metadata.get("classification_id"),
                    "task_name": observer.task.name,
                    "instruction": observer.task_description,
                    "category": task_metadata.get("category"),
                    "difficulty_level": task_metadata.get("difficulty_level"),
                    "episode_index": episode_index,
                    "init_state_id": init_state_id,
                    "num_init_states": observer.num_init_states,
                    "seed": args_payload["seed"],
                    "success": bool(success),
                    "failure_reason": None if success else "max_steps_exceeded",
                    "steps": steps,
                    "policy_steps": steps,
                    "initialization_steps": official["initialization_steps"],
                    "total_env_steps": official["initialization_steps"] + steps,
                    "max_policy_steps": horizon,
                    "elapsed_seconds": round(elapsed, 3),
                }
                results["episodes"].append(record)
                completed.add((shard["benchmark"], task_id, episode_index))
                results["updated_at"] = utc_now()
                results["summary"] = summarize(results["episodes"])
                atomic_json_dump(results, output)
                print(json.dumps(record, ensure_ascii=False), flush=True)
        finally:
            observer.close()

    results["finished_at"] = utc_now()
    results["updated_at"] = results["finished_at"]
    results["summary"] = summarize(results["episodes"])
    atomic_json_dump(results, output)


def env_worker_main(worker_index, task_queue, request_queue, response_queue, status_queue, args_payload):
    random.seed(args_payload["seed"])
    np.random.seed(args_payload["seed"])
    metadata = load_task_metadata(args_payload["libero_root"])
    try:
        while True:
            job = task_queue.get()
            if job is None:
                break
            evaluate_shard(job, args_payload, request_queue, response_queue, worker_index, metadata)
            status_queue.put(("complete", worker_index, job["shard"]["name"], None))
    except BaseException:
        status_queue.put(("error", worker_index, None, traceback.format_exc()))
        raise


def touch_prompt_lru(policy, task_description, limit):
    prompt = policy.default_prompt.format(task_prompt=task_description)
    cached = policy._prompt_cache.pop(prompt, None)
    if cached is not None:
        policy._prompt_cache[prompt] = cached
    while len(policy._prompt_cache) >= limit and prompt not in policy._prompt_cache:
        policy._prompt_cache.pop(next(iter(policy._prompt_cache)))


def build_protocol(args, policy, policy_config, official_protocol, implementation, selected_tasks):
    physical_device = (os.environ.get("CUDA_VISIBLE_DEVICES") or args.device.split(":", 1)[1]).split(",")[0]
    return {
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "adapter": resolved(args.adapter),
        "config": resolved(args.config),
        "model_path": resolved(args.model_path),
        "dataset_stats": resolved(args.dataset_stats),
        "libero_root": resolved(args.libero_root),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_cuda_device": physical_device,
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "seed": args.seed,
        "action_infer_steps": int(policy.action_infer_steps),
        "actions_per_plan": int(policy.actions_per_plan),
        "action_chunk_size": int(policy.action_chunk_size),
        "configured_num_steps_wait_ignored": int(policy_config.get("num_steps_wait", 0)),
        "prompt_cache_mode": "lazy-lru",
        "prompt_cache_limit": args.prompt_cache_limit,
        "text_encoder_released": False,
        "shared_policy_env_workers": args.env_workers,
        "official_evaluation": official_protocol,
        "fastwam_evaluation_implementation": implementation,
        "max_policy_steps": args.max_steps or official_protocol["max_policy_steps"],
        "render_size": args.render_size,
        "task_counts": {key: len(value) for key, value in selected_tasks.items()},
    }


def args_payload(args):
    return {
        "episodes_per_task": args.episodes_per_task,
        "episode_offset": args.episode_offset,
        "max_steps": args.max_steps,
        "render_size": args.render_size,
        "seed": args.seed,
        "libero_root": resolved(args.libero_root),
    }


def main():
    import eval_fastwam_libero as base_eval
    from libero_eval_protocol import load_official_evaluation_protocol

    args = parse_args()
    if args.episodes_per_task <= 0 or args.episode_offset < 0:
        raise ValueError("Episode count must be positive and offset must be non-negative")
    if args.tasks_per_shard <= 0 or args.env_workers <= 0 or args.prompt_cache_limit <= 0:
        raise ValueError("Shard size, env worker count, and prompt cache limit must be positive")
    if args.assignment_count <= 0 or not 0 <= args.assignment_index < args.assignment_count:
        raise ValueError("Invalid assignment index/count")
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = args.device.split(":", 1)[1]

    selected_tasks = discover_selected_tasks(args)
    all_shards = build_shards(selected_tasks, args.tasks_per_shard)
    assigned_shards = [
        shard for index, shard in enumerate(all_shards) if index % args.assignment_count == args.assignment_index
    ]
    output_root = Path(args.output_root).expanduser().resolve()
    shards_dir = output_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    official_protocol = load_official_evaluation_protocol(args.libero_root)
    implementation = evaluation_implementation()

    pending_shards = []
    for shard in assigned_shards:
        output = shards_dir / f"{shard['name']}.json"
        if not shard_is_complete(
            output,
            shard,
            args,
            official_protocol,
            implementation,
            args.expected_actions_per_plan,
        ):
            pending_shards.append(shard)
    print(
        f"[shared-policy] assigned_shards={len(assigned_shards)} "
        f"pending_shards={len(pending_shards)} env_workers={args.env_workers}",
        flush=True,
    )
    if not pending_shards:
        return

    policy, policy_config = base_eval.build_policy(args)
    if int(policy.action_infer_steps) != args.expected_action_infer_steps:
        policy.close()
        raise RuntimeError(
            f"Expected action_infer_steps={args.expected_action_infer_steps}, got {policy.action_infer_steps}"
        )
    if int(policy.actions_per_plan) != args.expected_actions_per_plan:
        policy.close()
        raise RuntimeError(
            f"Expected actions_per_plan={args.expected_actions_per_plan}, got {policy.actions_per_plan}"
        )
    protocol = build_protocol(args, policy, policy_config, official_protocol, implementation, selected_tasks)

    jobs = []
    for shard in pending_shards:
        output = shards_dir / f"{shard['name']}.json"
        signature = shard_run_signature(args, shard, official_protocol, implementation)
        signature["shared_policy"]["actions_per_plan"] = int(policy.actions_per_plan)
        jobs.append(
            {
                "shard": shard,
                "output": str(output),
                "run_signature": signature,
                "protocol": protocol,
            }
        )

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    request_queue = context.Queue()
    status_queue = context.Queue()
    response_queues = [context.Queue(maxsize=1) for _ in range(args.env_workers)]
    workers = [
        context.Process(
            target=env_worker_main,
            args=(index, task_queue, request_queue, response_queues[index], status_queue, args_payload(args)),
            name=f"libero-env-{index}",
        )
        for index in range(args.env_workers)
    ]
    for worker in workers:
        worker.start()
    for job in jobs:
        task_queue.put(job)
    for _ in workers:
        task_queue.put(None)

    if args.ready_file:
        atomic_json_dump(
            {
                "ready_at": utc_now(),
                "pid": os.getpid(),
                "output_root": str(output_root),
                "action_infer_steps": int(policy.action_infer_steps),
                "env_workers": args.env_workers,
                "text_encoder_released": False,
            },
            args.ready_file,
        )

    completed_shards = 0
    plan_count = 0
    started = time.monotonic()
    failure = None
    try:
        while any(worker.is_alive() for worker in workers):
            while True:
                try:
                    status, worker_index, shard_name, detail = status_queue.get_nowait()
                except queue.Empty:
                    break
                if status == "complete":
                    completed_shards += 1
                    print(
                        f"[complete] worker={worker_index} shard={shard_name} "
                        f"completed={completed_shards}/{len(jobs)}",
                        flush=True,
                    )
                else:
                    failure = RuntimeError(f"Environment worker {worker_index} failed:\n{detail}")
                    break
            if failure is not None:
                break
            try:
                worker_index, request_id, images, state, task_description = request_queue.get(timeout=0.1)
            except queue.Empty:
                for worker in workers:
                    if worker.exitcode not in (None, 0):
                        failure = RuntimeError(f"Environment worker {worker.name} exited with {worker.exitcode}")
                        break
                continue
            try:
                touch_prompt_lru(policy, task_description, args.prompt_cache_limit)
                actions = policy.predict_action_chunk(images, state, task_description)
                plan_count += 1
                response_queues[worker_index].put((request_id, actions, None))
            except BaseException:
                detail = traceback.format_exc()
                response_queues[worker_index].put((request_id, None, detail))
                failure = RuntimeError(f"Shared policy inference failed:\n{detail}")
                break
        if failure is not None:
            raise failure
        for worker in workers:
            worker.join()
            if worker.exitcode:
                raise RuntimeError(f"Environment worker {worker.name} exited with {worker.exitcode}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=10)
        policy.close()
        if args.ready_file:
            Path(args.ready_file).unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    summary = {
        "finished_at": utc_now(),
        "assignment_index": args.assignment_index,
        "assignment_count": args.assignment_count,
        "assigned_shards": len(assigned_shards),
        "evaluated_shards": len(jobs),
        "completed_shards": completed_shards,
        "action_plans": plan_count,
        "elapsed_seconds": round(elapsed, 3),
        "plans_per_second": round(plan_count / elapsed, 6) if elapsed else 0.0,
        "protocol": protocol,
    }
    atomic_json_dump(summary, output_root / f"shared-server-{args.assignment_index:02d}.json")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
