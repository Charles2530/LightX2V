"""Export and evaluate one FastWAM action-DMD checkpoint on LIBERO."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "lightx2v_train"
BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OFFICIAL_HORIZONS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
}


@dataclass(frozen=True)
class EvalShard:
    benchmark: str
    task_ids: tuple[int, ...]

    @property
    def name(self):
        return f"{self.benchmark}-tasks-{self.task_ids[0]}-{self.task_ids[-1]}"


def build_shards():
    return [EvalShard(benchmark, tuple(range(start, start + 5))) for benchmark in BENCHMARKS for start in (0, 5)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--libero-root")
    parser.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def atomic_json_dump(payload, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


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


def shard_is_complete(path, shard, episodes_per_task):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not payload.get("finished_at"):
        return False
    episodes = payload.get("episodes", [])
    expected = {(task_id, episode) for task_id in shard.task_ids for episode in range(episodes_per_task)}
    actual = {(int(item["task_id"]), int(item["episode_index"])) for item in episodes}
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
        str(Path(args.train_config).expanduser().resolve()),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(export_path),
    ]
    log_path = output_root / "export.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.devices[0])
    env["PYTHONPATH"] = os.pathsep.join([str(TRAIN_ROOT), str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
    if not export_path.is_file():
        raise RuntimeError(f"Export command succeeded without creating {export_path}")
    return export_path


def build_eval_command(args, adapter, output_path, shard, device):
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_fastwam_libero.py")),
        "--config",
        str(Path(args.policy_config).expanduser().resolve()),
        "--adapter",
        str(adapter),
        "--model-path",
        str(Path(args.model_path).expanduser().resolve()),
        "--dataset-stats",
        str(Path(args.dataset_stats).expanduser().resolve()),
        "--output",
        str(output_path),
        "--benchmarks",
        shard.benchmark,
        "--task-ids",
        *(str(task_id) for task_id in shard.task_ids),
        "--episodes-per-task",
        str(args.episodes_per_task),
        "--device",
        f"cuda:{device}",
    ]
    if args.libero_root:
        command.extend(["--libero-root", str(Path(args.libero_root).expanduser().resolve())])
    return command


def run_shards(args, adapter, output_root):
    shards_dir = output_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    pending = [shard for shard in build_shards() if not shard_is_complete(shards_dir / f"{shard.name}.json", shard, args.episodes_per_task)]
    devices = [int(device) for device in args.devices]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct CUDA device indices")

    active = {}
    failed = []
    while pending or active:
        free_devices = [device for device in devices if device not in active]
        for device in free_devices:
            if not pending:
                break
            shard = pending.pop(0)
            output_path = shards_dir / f"{shard.name}.json"
            log_path = shards_dir / f"{shard.name}.log"
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.update(
                {
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "MUJOCO_EGL_DEVICE_ID": "0",
                    "PYTHONPATH": os.pathsep.join([str(TRAIN_ROOT), str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
                }
            )
            process = subprocess.Popen(
                build_eval_command(args, adapter, output_path, shard, device),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            active[device] = (process, log, shard, log_path)
            print(f"[launch] device={device} shard={shard.name}", flush=True)

        time.sleep(1)
        for device, (process, log, shard, log_path) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log.close()
            del active[device]
            if return_code:
                failed.append((shard.name, return_code, log_path))
                print(f"[failed] device={device} shard={shard.name} code={return_code}", flush=True)
            else:
                print(f"[complete] device={device} shard={shard.name}", flush=True)
    if failed:
        details = ", ".join(f"{name} (code={code}, log={path})" for name, code, path in failed)
        raise RuntimeError(f"LIBERO evaluation shards failed: {details}")


def aggregate_results(output_root, checkpoint, adapter, iteration, episodes_per_task):
    records = {}
    shards_dir = output_root / "shards"
    for shard in build_shards():
        path = shards_dir / f"{shard.name}.json"
        if not shard_is_complete(path, shard, episodes_per_task):
            raise RuntimeError(f"Missing or incomplete result shard: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload["episodes"]:
            key = (episode["benchmark"], int(episode["task_id"]), int(episode["episode_index"]))
            records[key] = episode

    episodes = [records[key] for key in sorted(records)]
    expected_count = len(BENCHMARKS) * 10 * episodes_per_task
    if len(episodes) != expected_count:
        raise RuntimeError(f"Aggregated {len(episodes)} unique episodes, expected {expected_count}")

    benchmark_summary = {}
    for benchmark in BENCHMARKS:
        items = [episode for episode in episodes if episode["benchmark"] == benchmark]
        successes = sum(bool(episode["success"]) for episode in items)
        benchmark_summary[benchmark] = {
            "episodes": len(items),
            "successes": successes,
            "success_rate": successes / len(items),
        }
    successes = sum(bool(episode["success"]) for episode in episodes)
    result = {
        "checkpoint": str(checkpoint),
        "adapter": str(adapter),
        "checkpoint_iteration": iteration,
        "evaluation_protocol": f"{episodes_per_task}_episode_per_task",
        "official_horizons": OFFICIAL_HORIZONS,
        "wait_steps": 30,
        "render_size": 256,
        "summary": {
            "benchmarks": benchmark_summary,
            "overall": {
                "episodes": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes),
            },
        },
        "episodes": episodes,
    }
    atomic_json_dump(result, output_root / "summary.json")
    return result


def main():
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    checkpoint, iteration, world_size = validate_checkpoint(args.checkpoint)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] iteration={iteration} world_size={world_size} path={checkpoint}", flush=True)
    adapter = export_student(args, checkpoint, iteration, output_root)
    print(f"[export] adapter={adapter}", flush=True)
    run_shards(args, adapter, output_root)
    result = aggregate_results(output_root, checkpoint, adapter, iteration, args.episodes_per_task)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
