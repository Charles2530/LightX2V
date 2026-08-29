"""Launch and aggregate shared-policy FastWAM evaluation across CUDA devices."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import eval_fastwam_libero_shared_policy as shared_eval
from libero_eval_protocol import load_official_evaluation_protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(shared_eval.DEFAULT_BENCHMARKS))
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--devices", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--env-workers-per-device", type=int, default=32)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--tasks-per-shard", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-action-infer-steps", type=int, default=1)
    parser.add_argument("--expected-actions-per-plan", type=int, default=10)
    parser.add_argument("--prompt-cache-limit", type=int, default=256)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
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


def validate_args(args):
    if not Path(args.adapter).expanduser().is_file():
        raise FileNotFoundError(args.adapter)
    if len(set(args.devices)) != len(args.devices) or not args.devices:
        raise ValueError("--devices must contain distinct CUDA device indices")
    if (
        args.env_workers_per_device <= 0
        or args.episodes_per_task <= 0
        or args.episode_offset < 0
        or args.tasks_per_shard <= 0
        or args.prompt_cache_limit <= 0
        or args.startup_timeout <= 0
    ):
        raise ValueError("Worker, episode, shard, cache, and timeout values must be positive")
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if args.expected_action_infer_steps != 1:
        raise ValueError("This evaluation requires action_infer_steps=1")


def server_command(args, device_index, assignment_index, output_root, ready_path):
    command = [
        sys.executable,
        str(TOOLS_ROOT / "eval_fastwam_libero_shared_policy.py"),
        "--config",
        resolved(args.policy_config),
        "--adapter",
        resolved(args.adapter),
        "--model-path",
        resolved(args.model_path),
        "--dataset-stats",
        resolved(args.dataset_stats),
        "--output-root",
        str(output_root),
        "--libero-root",
        resolved(args.libero_root),
        "--benchmarks",
        *args.benchmarks,
        "--episodes-per-task",
        str(args.episodes_per_task),
        "--episode-offset",
        str(args.episode_offset),
        "--tasks-per-shard",
        str(args.tasks_per_shard),
        "--render-size",
        str(args.render_size),
        "--seed",
        str(args.seed),
        "--expected-action-infer-steps",
        str(args.expected_action_infer_steps),
        "--expected-actions-per-plan",
        str(args.expected_actions_per_plan),
        "--device",
        "cuda:0",
        "--env-workers",
        str(args.env_workers_per_device),
        "--prompt-cache-limit",
        str(args.prompt_cache_limit),
        "--assignment-index",
        str(assignment_index),
        "--assignment-count",
        str(len(args.devices)),
        "--ready-file",
        str(ready_path),
    ]
    if args.task_ids is not None:
        command.extend(["--task-ids", *(str(task_id) for task_id in args.task_ids)])
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    visible_devices = str(device_index) if device_index == 0 else f"{device_index},0"
    return command, visible_devices


def build_manifest(args, output_root, selected_tasks, shards, commands, official, implementation):
    return {
        "generated_at": utc_now(),
        "launcher_command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "evaluation_mode": "shared-policy-multiprocessing-queues-v1",
        "adapter": resolved(args.adapter),
        "model_path": resolved(args.model_path),
        "dataset_stats": resolved(args.dataset_stats),
        "policy_config": resolved(args.policy_config),
        "libero_root": resolved(args.libero_root),
        "benchmarks": list(args.benchmarks),
        "task_counts": {key: len(value) for key, value in selected_tasks.items()},
        "episodes_per_task": args.episodes_per_task,
        "episode_offset": args.episode_offset,
        "tasks_per_shard": args.tasks_per_shard,
        "expected_shards": len(shards),
        "max_steps_override": args.max_steps,
        "render_size": args.render_size,
        "seed": args.seed,
        "expected_action_infer_steps": args.expected_action_infer_steps,
        "expected_actions_per_plan": args.expected_actions_per_plan,
        "prompt_cache_mode": "lazy-lru",
        "prompt_cache_limit": args.prompt_cache_limit,
        "text_encoder_released": False,
        "devices": list(args.devices),
        "env_workers_per_device": args.env_workers_per_device,
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
        "launcher_implementation": {
            "lightx2v_train/tools/eval_fastwam_libero_shared_checkpoint.py": shared_eval.file_record(
                Path(__file__).resolve()
            )
        },
        "commands": commands,
        "output_root": str(output_root),
    }


def stop_servers(active):
    for item in active:
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    for item in active:
        process = item["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        item["log"].close()


def launch_servers(args, output_root, command_items):
    ready_dir = output_root / ".server_ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    active = []
    try:
        for item in command_items:
            ready_path = Path(item["ready_file"])
            ready_path.unlink(missing_ok=True)
            log_path = Path(item["log"])
            log = log_path.open("a", encoding="utf-8")
            log.write(f"\n[{utc_now()}] {item['command']}\n")
            log.flush()
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": item["cuda_visible_devices"],
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "MUJOCO_EGL_DEVICE_ID": "0",
                    "PYTHONPATH": os.pathsep.join(
                        [str(ROOT / "lightx2v_train"), str(ROOT), env.get("PYTHONPATH", "")]
                    ).rstrip(os.pathsep),
                }
            )
            process = subprocess.Popen(
                item["argv"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            active.append({**item, "process": process, "log": log})
            print(
                f"[launch] physical_device={item['physical_cuda_device']} pid={process.pid} "
                f"assignment={item['assignment_index']}/{len(command_items)}",
                flush=True,
            )
            deadline = time.monotonic() + args.startup_timeout
            while not ready_path.is_file():
                return_code = process.poll()
                if return_code is not None:
                    if return_code:
                        raise RuntimeError(
                            f"Shared policy server on CUDA {item['physical_cuda_device']} failed with "
                            f"code {return_code}; log={item['log']}"
                        )
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Shared policy server on CUDA {item['physical_cuda_device']} did not become ready; "
                        f"log={item['log']}"
                    )
                time.sleep(1)
            if ready_path.is_file():
                print(f"[ready] physical_device={item['physical_cuda_device']}", flush=True)

        unfinished = {item["process"].pid: item for item in active if item["process"].poll() is None}
        while unfinished:
            time.sleep(2)
            for pid, item in list(unfinished.items()):
                return_code = item["process"].poll()
                if return_code is None:
                    continue
                del unfinished[pid]
                item["log"].close()
                if return_code:
                    raise RuntimeError(
                        f"Shared policy server on CUDA {item['physical_cuda_device']} failed with "
                        f"code {return_code}; log={item['log']}"
                    )
                print(f"[complete] physical_device={item['physical_cuda_device']}", flush=True)
        return active
    except BaseException:
        stop_servers(active)
        raise


def aggregate_results(args, output_root, selected_tasks, shards, manifest, official, implementation, started_at):
    records = {}
    protocols = []
    shards_dir = output_root / "shards"
    for shard in shards:
        path = shards_dir / f"{shard['name']}.json"
        if not shared_eval.shard_is_complete(
            path,
            shard,
            args,
            official,
            implementation,
            args.expected_actions_per_plan,
        ):
            raise RuntimeError(f"Missing, incomplete, or mismatched result shard: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocols.append(payload["protocol"])
        for episode in payload["episodes"]:
            key = (episode["benchmark"], int(episode["task_id"]), int(episode["episode_index"]))
            if key in records:
                raise RuntimeError(f"Duplicate episode across shards: {key}")
            records[key] = episode

    episodes = [records[key] for key in sorted(records)]
    expected_count = sum(len(task_ids) for task_ids in selected_tasks.values()) * args.episodes_per_task
    if len(episodes) != expected_count:
        raise RuntimeError(f"Aggregated {len(episodes)} episodes, expected {expected_count}")
    physical_devices = {int(item["physical_cuda_device"]) for item in protocols}
    if physical_devices != set(args.devices):
        raise RuntimeError(f"Result shards used physical CUDA devices {sorted(physical_devices)}, expected {args.devices}")
    for item in protocols:
        if (
            int(item["action_infer_steps"]) != args.expected_action_infer_steps
            or int(item["actions_per_plan"]) != args.expected_actions_per_plan
            or int(item["seed"]) != args.seed
            or item["official_evaluation"] != official
            or item["fastwam_evaluation_implementation"] != implementation
            or item.get("prompt_cache_mode") != "lazy-lru"
            or item.get("text_encoder_released") is not False
        ):
            raise RuntimeError("At least one shard has a mismatched shared-policy protocol")
    for episode in episodes:
        if (
            int(episode["seed"]) != args.seed
            or int(episode["initialization_steps"]) != official["initialization_steps"]
            or int(episode["max_policy_steps"]) != (args.max_steps or official["max_policy_steps"])
            or int(episode["policy_steps"]) != int(episode["steps"])
            or int(episode["init_state_id"])
            != int(episode["episode_index"]) % int(episode["num_init_states"])
        ):
            raise RuntimeError(
                f"Episode violates seed, initial-state, or horizon rules: "
                f"{episode['benchmark']}/{episode['task_id']}/{episode['episode_index']}"
            )

    result = {
        "adapter": resolved(args.adapter),
        "evaluation_protocol": f"{args.episodes_per_task}_episodes_per_task",
        "evaluation_mode": manifest["evaluation_mode"],
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
        "max_policy_steps": args.max_steps or official["max_policy_steps"],
        "task_counts": {key: len(value) for key, value in selected_tasks.items()},
        "seed": args.seed,
        "action_infer_steps": args.expected_action_infer_steps,
        "actions_per_plan": args.expected_actions_per_plan,
        "render_size": args.render_size,
        "started_at": started_at,
        "finished_at": utc_now(),
        "protocols_verified": len(protocols),
        "summary": shared_eval.summarize(episodes),
        "episodes": episodes,
    }
    atomic_json_dump(result, output_root / "summary.json")
    return result


def main():
    args = parse_args()
    # Shared shard signatures use the single-worker evaluator's `config` name.
    args.config = args.policy_config
    validate_args(args)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    selected_tasks = shared_eval.discover_selected_tasks(args)
    shards = shared_eval.build_shards(selected_tasks, args.tasks_per_shard)
    official = load_official_evaluation_protocol(args.libero_root)
    implementation = shared_eval.evaluation_implementation()

    ready_dir = output_root / ".server_ready"
    command_items = []
    manifest_commands = []
    for assignment_index, device in enumerate(args.devices):
        ready_path = ready_dir / f"cuda-{device}.json"
        log_path = output_root / f"server-cuda-{device}.log"
        argv, visible_devices = server_command(args, device, assignment_index, output_root, ready_path)
        item = {
            "physical_cuda_device": device,
            "assignment_index": assignment_index,
            "cuda_visible_devices": visible_devices,
            "mujoco_egl_device_id": "0",
            "ready_file": str(ready_path),
            "log": str(log_path),
            "argv": argv,
            "command": shlex.join(argv),
        }
        command_items.append(item)
        manifest_commands.append({key: value for key, value in item.items() if key != "argv"})

    manifest = build_manifest(
        args,
        output_root,
        selected_tasks,
        shards,
        manifest_commands,
        official,
        implementation,
    )
    atomic_json_dump(manifest, output_root / "commands.json")
    print(
        f"[catalog] task_counts={manifest['task_counts']} shards={len(shards)} "
        f"servers={len(command_items)} env_workers_per_device={args.env_workers_per_device}",
        flush=True,
    )
    if args.dry_run:
        print(f"[dry-run] manifest={output_root / 'commands.json'}", flush=True)
        return

    active = launch_servers(args, output_root, command_items)
    for item in active:
        if not item["log"].closed:
            item["log"].close()
    result = aggregate_results(
        args,
        output_root,
        selected_tasks,
        shards,
        manifest,
        official,
        implementation,
        started_at,
    )
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
