"""Strictly verify and compare four shared-policy LIBERO-plus evaluations."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import aggregate_fastwam_libero_results as common


EVALUATION_MODE = "shared-policy-multiprocessing-queues-v1"
LOG_ERROR_MARKERS = (
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "ConnectionResetError",
    "BrokenPipeError",
    "Shared policy inference failed:",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--protocol-directory", default="official_protocol_shared_policy")
    parser.add_argument(
        "--weight",
        action="append",
        nargs=3,
        metavar=("LABEL", "RESULT_DIRECTORY", "ADAPTER"),
        required=True,
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    return parser.parse_args()


def parse_timestamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_evidence(adapter):
    path = Path(adapter)
    if not path.is_file():
        raise FileNotFoundError(f"Missing adapter checkpoint: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_server_logs(manifest):
    evidence = []
    for item in manifest["commands"]:
        path = Path(item["log"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing policy-server log: {path}")
        expected_command = item["command"]
        launch_lines = []
        errors = []
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if "eval_fastwam_libero_shared_policy.py" in line and line.startswith("["):
                    _, separator, command = line.partition("] ")
                    if not separator or command != expected_command:
                        raise RuntimeError(
                            f"Policy-server launch in {path}:{line_number} differs from commands.json"
                        )
                    required = (
                        "--expected-action-infer-steps 1",
                        "--expected-actions-per-plan 10",
                        "--seed 0",
                    )
                    missing = [token for token in required if token not in command]
                    if missing:
                        raise RuntimeError(
                            f"Policy-server log {path}:{line_number} is missing {missing}"
                        )
                    launch_lines.append(line_number)
                if any(marker in line for marker in LOG_ERROR_MARKERS) or (
                    "Environment worker " in line and " failed:" in line
                ):
                    errors.append({"line": line_number, "text": line[:500]})
        if not launch_lines:
            raise RuntimeError(f"Policy-server log has no recorded launch command: {path}")
        if errors:
            raise RuntimeError(f"Policy-server log contains error markers: {path}: {errors[:3]}")
        evidence.append(
            {
                "physical_cuda_device": int(item["physical_cuda_device"]),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "launch_count": len(launch_lines),
                "launch_line_numbers": launch_lines,
                "commands_match_manifest": True,
                "action_infer_steps": common.ACTION_INFER_STEPS,
                "actions_per_plan": 10,
                "seed": common.SEED,
                "error_markers": 0,
            }
        )
    return evidence


def validate_shards(protocol_root, expected_shards, official, implementation):
    paths = sorted((protocol_root / "shards").glob("*.json"))
    if len(paths) != expected_shards:
        raise RuntimeError(f"{protocol_root} has {len(paths)} shards, expected {expected_shards}")
    starts = []
    finishes = []
    physical_devices = set()
    for path in paths:
        payload = common.load_json(path)
        if not payload.get("finished_at"):
            raise RuntimeError(f"Incomplete shard: {path}")
        protocol = payload.get("protocol", {})
        if (
            int(protocol.get("action_infer_steps", -1)) != common.ACTION_INFER_STEPS
            or int(protocol.get("actions_per_plan", -1)) != 10
            or int(protocol.get("seed", -1)) != common.SEED
            or protocol.get("prompt_cache_mode") != "lazy-lru"
            or protocol.get("text_encoder_released") is not False
            or protocol.get("official_evaluation") != official
            or protocol.get("fastwam_evaluation_implementation") != implementation
        ):
            raise RuntimeError(f"Shared-policy protocol mismatch in {path}")
        physical_devices.add(int(protocol["physical_cuda_device"]))
        starts.append(parse_timestamp(payload["started_at"]))
        finishes.append(parse_timestamp(payload["finished_at"]))
    if physical_devices != {1, 2, 3, 4, 5, 6, 7}:
        raise RuntimeError(f"Shards used physical CUDA devices {sorted(physical_devices)}, expected 1-7")
    started = min(starts)
    finished = max(finishes)
    return {
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": (finished - started).total_seconds(),
    }


def validate_weight(label, result_dir, expected_adapter, reference, protocol_directory):
    protocol_root = result_dir / protocol_directory
    summary_path = protocol_root / "summary.json"
    commands_path = protocol_root / "commands.json"
    if not summary_path.is_file() or not commands_path.is_file():
        raise FileNotFoundError(f"Missing completed outputs for {label}: {summary_path}, {commands_path}")
    result = common.load_json(summary_path)
    manifest = common.load_json(commands_path)
    adapter = common.resolved(expected_adapter)

    if result.get("adapter") != adapter or manifest.get("adapter") != adapter:
        raise RuntimeError(f"{label} did not use expected adapter {adapter}")
    if result.get("evaluation_mode") != EVALUATION_MODE or manifest.get("evaluation_mode") != EVALUATION_MODE:
        raise RuntimeError(f"{label} is not a shared-policy evaluation")
    if (
        result.get("action_infer_steps") != common.ACTION_INFER_STEPS
        or result.get("actions_per_plan") != 10
        or result.get("seed") != common.SEED
        or result.get("max_policy_steps") != common.MAX_POLICY_STEPS
        or result.get("evaluation_protocol") != f"{common.EPISODES_PER_TASK}_episodes_per_task"
        or not result.get("finished_at")
    ):
        raise RuntimeError(f"{label} summary has the wrong rollout configuration")
    if (
        manifest.get("episodes_per_task") != common.EPISODES_PER_TASK
        or manifest.get("seed") != common.SEED
        or manifest.get("expected_action_infer_steps") != common.ACTION_INFER_STEPS
        or manifest.get("expected_actions_per_plan") != 10
        or manifest.get("devices") != [1, 2, 3, 4, 5, 6, 7]
        or manifest.get("prompt_cache_mode") != "lazy-lru"
        or manifest.get("text_encoder_released") is not False
    ):
        raise RuntimeError(f"{label} command manifest has the wrong shared-policy configuration")

    task_counts = result.get("task_counts")
    if set(task_counts or {}) != set(common.BENCHMARKS) or manifest.get("task_counts") != task_counts:
        raise RuntimeError(f"{label} suite catalog is incomplete or inconsistent")
    expected_tasks = sum(task_counts.values())
    expected_episodes = expected_tasks * common.EPISODES_PER_TASK
    expected_shards = int(manifest.get("expected_shards", -1))
    if expected_shards != expected_tasks or result.get("protocols_verified") != expected_shards:
        raise RuntimeError(f"{label} did not verify exactly one shard per task")
    if len(manifest.get("commands", [])) != 7:
        raise RuntimeError(f"{label} must record exactly seven physical GPU server commands")

    official = result.get("official_evaluation")
    implementation = result.get("fastwam_evaluation_implementation")
    if official != manifest.get("official_evaluation") or implementation != manifest.get(
        "fastwam_evaluation_implementation"
    ):
        raise RuntimeError(f"{label} protocol or implementation differs between summary and manifest")
    if (
        official.get("initialization_steps") != common.INITIALIZATION_STEPS
        or official.get("initialization_action") != [0.0] * 7
        or official.get("max_policy_steps") != common.MAX_POLICY_STEPS
    ):
        raise RuntimeError(f"{label} does not match official rollout mechanics")
    if reference is not None and (
        task_counts != reference["task_counts"]
        or official != reference["official_evaluation"]
        or implementation != reference["implementation"]
    ):
        raise RuntimeError(f"{label} catalog, protocol, or implementation differs from native")

    episodes = result.get("episodes", [])
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"{label} has {len(episodes)} episodes, expected {expected_episodes}")
    episode_keys = set()
    task_trials = {}
    initial_states = {} if reference is None else reference["initial_states"]
    for episode in episodes:
        key = (episode["benchmark"], int(episode["task_id"]), int(episode["episode_index"]))
        if key in episode_keys:
            raise RuntimeError(f"{label} contains duplicate episode {key}")
        episode_keys.add(key)
        task_trials[key[:2]] = task_trials.get(key[:2], 0) + 1
        state = (
            int(episode["init_state_id"]),
            int(episode["num_init_states"]),
            int(episode["seed"]),
        )
        if (
            state[0] != key[2] % state[1]
            or state[2] != common.SEED
            or int(episode["initialization_steps"]) != common.INITIALIZATION_STEPS
            or int(episode["max_policy_steps"]) != common.MAX_POLICY_STEPS
            or int(episode["policy_steps"]) != int(episode["steps"])
        ):
            raise RuntimeError(f"{label} episode {key} violates official rollout mechanics")
        if reference is None:
            initial_states[key] = state
        elif initial_states.get(key) != state:
            raise RuntimeError(f"{label} episode {key} has a different seed or initial state")
    if set(episode["benchmark"] for episode in episodes) != set(common.BENCHMARKS):
        raise RuntimeError(f"{label} does not contain all LIBERO-plus suites")
    invalid_trials = {key: count for key, count in task_trials.items() if count != common.EPISODES_PER_TASK}
    if invalid_trials or len(task_trials) != expected_tasks:
        raise RuntimeError(f"{label} does not contain exactly 50 trials per task")
    if reference is not None and episode_keys != set(initial_states):
        raise RuntimeError(f"{label} episode catalog differs from native")

    wall_clock = validate_shards(protocol_root, expected_shards, official, implementation)
    logs = validate_server_logs(manifest)
    return {
        "reference": {
            "task_counts": task_counts,
            "official_evaluation": official,
            "implementation": implementation,
            "initial_states": initial_states,
        },
        "output": {
            "label": label,
            "adapter": adapter,
            "checkpoint": checkpoint_evidence(adapter),
            "summary_json": str(summary_path.resolve()),
            "commands_json": str(commands_path.resolve()),
            "launcher_command": manifest.get("launcher_command"),
            "server_commands": [item.get("command") for item in manifest["commands"]],
            "server_log_evidence": logs,
            "model_path": common.resolved(manifest["model_path"]),
            "dataset_stats": common.resolved(manifest["dataset_stats"]),
            "policy_config": common.resolved(manifest["policy_config"]),
            "libero_root": common.resolved(manifest["libero_root"]),
            "task_counts": task_counts,
            "episodes_per_task": common.EPISODES_PER_TASK,
            "total_episodes": expected_episodes,
            "seed": common.SEED,
            "action_infer_steps": common.ACTION_INFER_STEPS,
            "prompt_cache_mode": "lazy-lru-with-retained-text-encoder",
            "official_evaluation": official,
            "wall_clock": wall_clock,
            "summary": common.summarize(episodes),
        },
    }


def main():
    args = parse_args()
    if len(args.weight) != 4:
        raise ValueError("Exactly four ordered --weight entries are required")
    results_root = Path(args.results_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else results_root / "comparison_summary.json"
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else results_root / "comparison_summary.csv"
    weights = {}
    reference = None
    for label, directory, adapter in args.weight:
        if label in weights:
            raise ValueError(f"Duplicate weight label: {label}")
        result_dir = Path(directory)
        if not result_dir.is_absolute():
            result_dir = results_root / result_dir
        validated = validate_weight(
            label,
            result_dir.resolve(),
            adapter,
            reference,
            args.protocol_directory,
        )
        if reference is None:
            reference = validated["reference"]
        weights[label] = validated["output"]

    payload = {
        "generated_at": common.utc_now(),
        "results_root": str(results_root),
        "verification": {
            "weights_verified": list(weights),
            "all_suites_and_tasks_complete": True,
            "same_episode_catalog_seed_and_initial_states": True,
            "official_protocol_and_implementation_match": True,
            "shared_policy_lazy_prompt_cache_verified": True,
            "physical_cuda_devices": [1, 2, 3, 4, 5, 6, 7],
            "episodes_per_task": common.EPISODES_PER_TASK,
            "total_tasks_per_weight": sum(reference["task_counts"].values()),
            "total_episodes_per_weight": sum(reference["task_counts"].values())
            * common.EPISODES_PER_TASK,
        },
        "weights": weights,
    }
    common.atomic_json_dump(payload, output_json)
    common.atomic_csv_dump(weights, output_csv)
    print(f"[verified] json={output_json} csv={output_csv}")


if __name__ == "__main__":
    main()
