"""Export if needed, then evaluate one FastWAM adapter on LIBERO or LIBERO-plus."""

import argparse
import gc
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT / "lightx2v_train"
SIMULATOR_SRC = ROOT / "lightx2v_ros" / "src" / "simulator"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from libero_eval_protocol import (
    load_fastwam_evaluation_implementation,
    load_official_evaluation_protocol,
)

DEFAULT_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class EvalShard:
    benchmark: str
    task_ids: tuple[int, ...]

    @property
    def name(self):
        return f"{self.benchmark}-tasks-{self.task_ids[0]:05d}-{self.task_ids[-1]:05d}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", help="Already-exported native FastWAM adapter")
    source.add_argument("--checkpoint", help="Action-DMD training checkpoint directory to export")
    parser.add_argument("--train-config", help="Required with --checkpoint")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=1,
        help="Independent evaluation workers to colocate on each CUDA device",
    )
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--tasks-per-shard", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-action-infer-steps", type=int, default=1)
    parser.add_argument("--release-text-encoder-after-prompt-cache", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write command manifests without launching shards")
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


def discover_task_counts(libero_root, benchmarks):
    if str(SIMULATOR_SRC) not in sys.path:
        sys.path.insert(0, str(SIMULATOR_SRC))
    from simulator.libero_node.observer import create_task_suite, load_libero

    benchmark_module, _, _ = load_libero(Path(libero_root).expanduser().resolve())
    factories = benchmark_module.get_benchmark_dict()
    counts = {}
    for benchmark in benchmarks:
        if benchmark not in factories:
            raise ValueError(f"Benchmark {benchmark!r} is not registered by {libero_root}")
        counts[benchmark] = create_task_suite(factories[benchmark]).get_num_tasks()
    return counts


def build_shards(task_counts, benchmarks, tasks_per_shard):
    shards = []
    for benchmark in benchmarks:
        for start in range(0, task_counts[benchmark], tasks_per_shard):
            stop = min(start + tasks_per_shard, task_counts[benchmark])
            shards.append(EvalShard(benchmark, tuple(range(start, stop))))
    return shards


def validate_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    required = ("training_state.pt", "student_action.pt", "fake_action.pt", "config.yaml")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {checkpoint}: missing {missing}")

    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    iteration = int(state["iteration"])
    world_size = int(state["world_size"])
    rng_files = sorted(checkpoint.glob("rng-rank-*.pt"))
    if len(rng_files) != world_size:
        raise RuntimeError(f"Checkpoint {checkpoint} has {len(rng_files)} RNG files, expected {world_size}")
    if checkpoint.name != f"checkpoint-{iteration:09d}":
        raise RuntimeError(f"Checkpoint directory {checkpoint.name!r} disagrees with iteration {iteration}")
    del state
    gc.collect()
    return checkpoint, iteration, world_size


def expected_shard_signature(args, adapter, shard):
    official_protocol = load_official_evaluation_protocol(args.libero_root)
    implementation = load_fastwam_evaluation_implementation(ROOT)
    return {
        "adapter": str(adapter),
        "config": resolved(args.policy_config),
        "model_path": resolved(args.model_path),
        "dataset_stats": resolved(args.dataset_stats),
        "libero_root": resolved(args.libero_root),
        "benchmarks": [shard.benchmark],
        "task_ids": {shard.benchmark: list(shard.task_ids)},
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


def shard_is_complete(path, shard, args, adapter):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not payload.get("finished_at"):
        return False
    if payload.get("run_signature") != expected_shard_signature(args, adapter, shard):
        return False
    protocol = payload.get("protocol", {})
    if (
        protocol.get("adapter") != str(adapter)
        or int(protocol.get("seed", -1)) != args.seed
        or int(protocol.get("action_infer_steps", -1)) != args.expected_action_infer_steps
    ):
        return False
    expected = {
        (shard.benchmark, task_id, episode)
        for task_id in shard.task_ids
        for episode in range(args.episode_offset, args.episode_offset + args.episodes_per_task)
    }
    episodes = payload.get("episodes", [])
    actual = {
        (item["benchmark"], int(item["task_id"]), int(item["episode_index"])) for item in episodes
    }
    return actual == expected


def export_student(args, checkpoint, iteration, output_root):
    export_path = checkpoint.parent / "exports" / f"checkpoint-{iteration:09d}-student.pt"
    if export_path.is_file() and not args.force_export:
        return export_path
    export_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("export_fastwam_action_dmd.py")),
        "--config",
        resolved(args.train_config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(export_path),
    ]
    log_path = output_root / "export.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.devices[0])
    env["PYTHONPATH"] = os.pathsep.join([str(TRAIN_ROOT), str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {shlex.join(command)}\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
    if not export_path.is_file():
        raise RuntimeError(f"Export command succeeded without creating {export_path}")
    return export_path


def build_eval_command(args, adapter, output_path, shard, device, ready_path):
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_fastwam_libero.py")),
        "--config",
        resolved(args.policy_config),
        "--adapter",
        str(adapter),
        "--model-path",
        resolved(args.model_path),
        "--dataset-stats",
        resolved(args.dataset_stats),
        "--output",
        str(output_path),
        "--libero-root",
        resolved(args.libero_root),
        "--benchmarks",
        shard.benchmark,
        "--task-ids",
        *(str(task_id) for task_id in shard.task_ids),
        "--episodes-per-task",
        str(args.episodes_per_task),
        "--episode-offset",
        str(args.episode_offset),
        "--render-size",
        str(args.render_size),
        "--seed",
        str(args.seed),
        "--expected-action-infer-steps",
        str(args.expected_action_infer_steps),
        "--device",
        "cuda:0",
        "--ready-file",
        str(ready_path),
    ]
    if args.release_text_encoder_after_prompt_cache:
        command.append("--release-text-encoder-after-prompt-cache")
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    return command


def write_command_manifest(args, adapter, output_root, task_counts, shards):
    official_protocol = load_official_evaluation_protocol(args.libero_root)
    implementation = load_fastwam_evaluation_implementation(ROOT)
    worker_slots = [
        (device, worker_index)
        for worker_index in range(args.workers_per_device)
        for device in args.devices
    ]
    commands = []
    for index, shard in enumerate(shards):
        device, worker_index = worker_slots[index % len(worker_slots)]
        visible_devices = str(device) if device == 0 else f"{device},0"
        output = output_root / "shards" / f"{shard.name}.json"
        ready_path = output_root / ".worker_ready" / f"{shard.name}.json"
        commands.append(
            {
                "shard": shard.name,
                "physical_cuda_device": device,
                "worker_index": worker_index,
                "cuda_visible_devices": visible_devices,
                "mujoco_egl_device_id": "0",
                "command": shlex.join(build_eval_command(args, adapter, output, shard, device, ready_path)),
            }
        )
    manifest = {
        "generated_at": utc_now(),
        "launcher_command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "adapter": str(adapter),
        "model_path": resolved(args.model_path),
        "dataset_stats": resolved(args.dataset_stats),
        "policy_config": resolved(args.policy_config),
        "libero_root": resolved(args.libero_root),
        "benchmarks": args.benchmarks,
        "task_counts": task_counts,
        "episodes_per_task": args.episodes_per_task,
        "episode_offset": args.episode_offset,
        "tasks_per_shard": args.tasks_per_shard,
        "max_steps_override": args.max_steps,
        "render_size": args.render_size,
        "seed": args.seed,
        "expected_action_infer_steps": args.expected_action_infer_steps,
        "release_text_encoder_after_prompt_cache": args.release_text_encoder_after_prompt_cache,
        "official_evaluation": official_protocol,
        "fastwam_evaluation_implementation": implementation,
        "devices": args.devices,
        "workers_per_device": args.workers_per_device,
        "commands": commands,
    }
    atomic_json_dump(manifest, output_root / "commands.json")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in commands:
        lines.append(
            f"CUDA_VISIBLE_DEVICES={item['cuda_visible_devices']} "
            f"MUJOCO_EGL_DEVICE_ID={item['mujoco_egl_device_id']} {item['command']}"
        )
    (output_root / "commands.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def stop_active_processes(active):
    for process, _, _, _, _ in active.values():
        if process.poll() is None:
            process.terminate()
    for process, log, _, _, ready_path in active.values():
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()
        ready_path.unlink(missing_ok=True)


def run_shards(args, adapter, output_root, shards):
    shards_dir = output_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    ready_dir = output_root / ".worker_ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        shard
        for shard in shards
        if not shard_is_complete(shards_dir / f"{shard.name}.json", shard, args, adapter)
    ]
    devices = [int(device) for device in args.devices]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct CUDA device indices")
    worker_slots = [
        (device, worker_index)
        for worker_index in range(args.workers_per_device)
        for device in devices
    ]

    active = {}
    try:
        while pending or active:
            loading_devices = {
                device
                for (device, _), (process, _, _, _, ready_path) in active.items()
                if process.poll() is None and not ready_path.is_file()
            }
            launch_slots = []
            for device in devices:
                if device in loading_devices:
                    continue
                free_slots = [slot for slot in worker_slots if slot[0] == device and slot not in active]
                if free_slots:
                    launch_slots.append(free_slots[0])
            for device, worker_index in launch_slots:
                if not pending:
                    break
                shard = pending.pop(0)
                output_path = shards_dir / f"{shard.name}.json"
                log_path = shards_dir / f"{shard.name}.log"
                ready_path = ready_dir / f"{shard.name}.json"
                ready_path.unlink(missing_ok=True)
                command = build_eval_command(args, adapter, output_path, shard, device, ready_path)
                visible_devices = str(device) if device == 0 else f"{device},0"
                log = log_path.open("a", encoding="utf-8")
                log.write(
                    f"\n[{utc_now()}] CUDA_VISIBLE_DEVICES={visible_devices} "
                    f"MUJOCO_EGL_DEVICE_ID=0 {shlex.join(command)}\n"
                )
                log.flush()
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": visible_devices,
                        "MUJOCO_GL": "egl",
                        "PYOPENGL_PLATFORM": "egl",
                        "MUJOCO_EGL_DEVICE_ID": "0",
                        "PYTHONPATH": os.pathsep.join([str(TRAIN_ROOT), str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
                    }
                )
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                active[(device, worker_index)] = (process, log, shard, log_path, ready_path)
                print(
                    f"[launch] physical_device={device} worker={worker_index} "
                    f"shard={shard.name} pid={process.pid}",
                    flush=True,
                )

            time.sleep(1)
            for slot, (process, log, shard, log_path, ready_path) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log.close()
                ready_path.unlink(missing_ok=True)
                del active[slot]
                if return_code:
                    raise RuntimeError(
                        f"LIBERO evaluation shard {shard.name} failed with code {return_code}; log={log_path}"
                    )
                device, worker_index = slot
                print(
                    f"[complete] physical_device={device} worker={worker_index} shard={shard.name}",
                    flush=True,
                )
    finally:
        stop_active_processes(active)


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
    task_examples = {}
    for episode in episodes:
        task_examples.setdefault(f"{episode['benchmark']}/{episode['task_id']}", episode)
    for task_key, task_score in tasks.items():
        example = task_examples[task_key]
        task_score.update(
            {
                "task_name": example["task_name"],
                "category": example.get("category"),
                "difficulty_level": example.get("difficulty_level"),
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


def aggregate_results(args, output_root, checkpoint, adapter, iteration, task_counts, shards, started_at):
    records = {}
    protocols = []
    shards_dir = output_root / "shards"
    for shard in shards:
        path = shards_dir / f"{shard.name}.json"
        if not shard_is_complete(path, shard, args, adapter):
            raise RuntimeError(f"Missing or incomplete result shard: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocols.append(payload["protocol"])
        for episode in payload["episodes"]:
            key = (episode["benchmark"], int(episode["task_id"]), int(episode["episode_index"]))
            if key in records:
                raise RuntimeError(f"Duplicate episode across shards: {key}")
            records[key] = episode

    episodes = [records[key] for key in sorted(records)]
    expected_count = sum(task_counts.values()) * args.episodes_per_task
    if len(episodes) != expected_count:
        raise RuntimeError(f"Aggregated {len(episodes)} unique episodes, expected {expected_count}")
    if any(int(item["action_infer_steps"]) != args.expected_action_infer_steps for item in protocols):
        raise RuntimeError("Not every shard confirmed the requested action_infer_steps")
    if any(int(item["seed"]) != args.seed for item in protocols):
        raise RuntimeError("Not every shard confirmed the requested seed")
    official_protocol = load_official_evaluation_protocol(args.libero_root)
    implementation = load_fastwam_evaluation_implementation(ROOT)
    if any(item.get("official_evaluation") != official_protocol for item in protocols):
        raise RuntimeError("Not every shard used the current LIBERO-plus official evaluation protocol")
    if any(item.get("fastwam_evaluation_implementation") != implementation for item in protocols):
        raise RuntimeError("Not every shard used the current FastWAM evaluation implementation")

    result = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "adapter": str(adapter),
        "checkpoint_iteration": iteration,
        "evaluation_protocol": f"{args.episodes_per_task}_episodes_per_task",
        "official_evaluation": official_protocol,
        "fastwam_evaluation_implementation": implementation,
        "max_policy_steps": args.max_steps or official_protocol["max_policy_steps"],
        "task_counts": task_counts,
        "seed": args.seed,
        "action_infer_steps": args.expected_action_infer_steps,
        "render_size": args.render_size,
        "started_at": started_at,
        "finished_at": utc_now(),
        "protocols_verified": len(protocols),
        "summary": summarize(episodes),
        "episodes": episodes,
    }
    atomic_json_dump(result, output_root / "summary.json")
    return result


def main():
    args = parse_args()
    if (
        args.episodes_per_task <= 0
        or args.tasks_per_shard <= 0
        or args.workers_per_device <= 0
        or args.episode_offset < 0
    ):
        raise ValueError(
            "Episode count, shard size, and workers per device must be positive; "
            "offset must be non-negative"
        )
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if args.checkpoint and not args.train_config:
        raise ValueError("--train-config is required with --checkpoint")
    devices = [int(device) for device in args.devices]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct CUDA device indices")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    checkpoint = None
    iteration = None
    if args.checkpoint:
        checkpoint, iteration, world_size = validate_checkpoint(args.checkpoint)
        print(f"[checkpoint] iteration={iteration} world_size={world_size} path={checkpoint}", flush=True)
        adapter = export_student(args, checkpoint, iteration, output_root)
    else:
        adapter = Path(args.adapter).expanduser().resolve()
        if not adapter.is_file():
            raise FileNotFoundError(adapter)
    print(f"[adapter] {adapter}", flush=True)

    task_counts = discover_task_counts(args.libero_root, args.benchmarks)
    shards = build_shards(task_counts, args.benchmarks, args.tasks_per_shard)
    write_command_manifest(args, adapter, output_root, task_counts, shards)
    print(f"[catalog] task_counts={task_counts} shards={len(shards)}", flush=True)
    if args.dry_run:
        print(f"[dry-run] manifests written to {output_root}", flush=True)
        return
    run_shards(args, adapter, output_root, shards)
    result = aggregate_results(args, output_root, checkpoint, adapter, iteration, task_counts, shards, started_at)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
