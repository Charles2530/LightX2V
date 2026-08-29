"""Verify and compare completed FastWAM LIBERO-plus evaluations."""

import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EPISODES_PER_TASK = 50
SEED = 0
ACTION_INFER_STEPS = 1
INITIALIZATION_STEPS = 5
MAX_POLICY_STEPS = 600


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--weight",
        action="append",
        nargs=3,
        metavar=("LABEL", "RESULT_DIRECTORY", "ADAPTER"),
        required=True,
        help="Repeat for each result set in comparison order",
    )
    parser.add_argument("--output-json", help="Default: RESULTS_ROOT/comparison_summary.json")
    parser.add_argument("--output-csv", help="Default: RESULTS_ROOT/comparison_summary.csv")
    return parser.parse_args()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolved(path):
    return str(Path(path).expanduser().resolve())


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(payload, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def score(items):
    successes = sum(bool(item["success"]) for item in items)
    policy_steps = [int(item["policy_steps"]) for item in items]
    success_steps = [int(item["policy_steps"]) for item in items if item["success"]]
    elapsed = [float(item["elapsed_seconds"]) for item in items]
    failures = Counter(
        item.get("failure_reason") or "unknown" for item in items if not item["success"]
    )
    return {
        "episodes": len(items),
        "successes": successes,
        "failures": len(items) - successes,
        "success_rate": successes / len(items) if items else 0.0,
        "average_policy_steps": sum(policy_steps) / len(policy_steps) if policy_steps else 0.0,
        "average_success_policy_steps": (
            sum(success_steps) / len(success_steps) if success_steps else None
        ),
        "rollout_elapsed_seconds": round(sum(elapsed), 3),
        "average_rollout_elapsed_seconds": (
            round(sum(elapsed) / len(elapsed), 3) if elapsed else 0.0
        ),
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
        example = examples[task_key]
        task_score.update(
            {
                "task_name": example["task_name"],
                "category": example.get("category"),
                "difficulty_level": example.get("difficulty_level"),
            }
        )
    return {
        "overall": score(episodes),
        "suites": grouped_scores(episodes, lambda item: item["benchmark"]),
        "tasks": tasks,
    }


def parse_timestamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def shard_wall_time(protocol_root, expected_shards):
    shards = sorted((protocol_root / "shards").glob("*.json"))
    if len(shards) != expected_shards:
        raise RuntimeError(
            f"{protocol_root} has {len(shards)} result shards, expected {expected_shards}"
        )
    starts = []
    finishes = []
    for path in shards:
        payload = load_json(path)
        if not payload.get("finished_at"):
            raise RuntimeError(f"Incomplete result shard: {path}")
        starts.append(parse_timestamp(payload["started_at"]))
        finishes.append(parse_timestamp(payload["finished_at"]))
    started = min(starts)
    finished = max(finishes)
    return {
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": (finished - started).total_seconds(),
    }


def validate_weight(label, result_dir, expected_adapter, reference):
    protocol_root = result_dir / "official_protocol"
    summary_path = protocol_root / "summary.json"
    commands_path = protocol_root / "commands.json"
    if not summary_path.is_file() or not commands_path.is_file():
        raise FileNotFoundError(f"Missing completed outputs for {label}: {summary_path}, {commands_path}")

    result = load_json(summary_path)
    manifest = load_json(commands_path)
    adapter = resolved(expected_adapter)
    if result.get("adapter") != adapter or manifest.get("adapter") != adapter:
        raise RuntimeError(f"{label} did not use expected adapter {adapter}")
    if result.get("action_infer_steps") != ACTION_INFER_STEPS:
        raise RuntimeError(f"{label} action_infer_steps is not {ACTION_INFER_STEPS}")
    if result.get("seed") != SEED:
        raise RuntimeError(f"{label} seed is not {SEED}")
    if result.get("max_policy_steps") != MAX_POLICY_STEPS:
        raise RuntimeError(f"{label} max policy steps is not {MAX_POLICY_STEPS}")
    if result.get("evaluation_protocol") != f"{EPISODES_PER_TASK}_episodes_per_task":
        raise RuntimeError(f"{label} did not record {EPISODES_PER_TASK} episodes per task")
    if not result.get("finished_at"):
        raise RuntimeError(f"{label} summary is not finished")

    task_counts = result.get("task_counts")
    if set(task_counts or {}) != set(BENCHMARKS) or manifest.get("task_counts") != task_counts:
        raise RuntimeError(f"{label} suite catalog is incomplete or inconsistent")
    expected_episodes = sum(task_counts.values()) * EPISODES_PER_TASK
    commands = manifest.get("commands", [])
    if result.get("protocols_verified") != len(commands):
        raise RuntimeError(f"{label} did not verify every command shard")
    if manifest.get("episodes_per_task") != EPISODES_PER_TASK:
        raise RuntimeError(f"{label} command manifest has the wrong trial count")
    if manifest.get("seed") != SEED or manifest.get("expected_action_infer_steps") != ACTION_INFER_STEPS:
        raise RuntimeError(f"{label} command manifest has the wrong seed or inference steps")
    if manifest.get("devices") != [1, 2, 3, 4, 5, 6, 7]:
        raise RuntimeError(f"{label} command manifest did not use CUDA devices 1-7")

    official = result.get("official_evaluation")
    if official != manifest.get("official_evaluation"):
        raise RuntimeError(f"{label} official protocol differs between summary and manifest")
    if (
        official.get("initialization_steps") != INITIALIZATION_STEPS
        or official.get("initialization_action") != [0.0] * 7
        or official.get("max_policy_steps") != MAX_POLICY_STEPS
    ):
        raise RuntimeError(f"{label} does not match the official rollout mechanics")
    implementation = result.get("fastwam_evaluation_implementation")
    if implementation != manifest.get("fastwam_evaluation_implementation"):
        raise RuntimeError(f"{label} implementation fingerprint differs between outputs")

    if reference is not None:
        if task_counts != reference["task_counts"]:
            raise RuntimeError(f"{label} task counts differ from the native evaluation")
        if official != reference["official_evaluation"]:
            raise RuntimeError(f"{label} official protocol differs from the native evaluation")
        if implementation != reference["implementation"]:
            raise RuntimeError(f"{label} evaluator implementation differs from the native evaluation")

    episodes = result.get("episodes", [])
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"{label} has {len(episodes)} episodes, expected {expected_episodes}")
    episode_keys = set()
    initial_states = {} if reference is None else reference["initial_states"]
    task_trials = Counter()
    for episode in episodes:
        key = (
            episode["benchmark"],
            int(episode["task_id"]),
            int(episode["episode_index"]),
        )
        if key in episode_keys:
            raise RuntimeError(f"{label} contains duplicate episode {key}")
        episode_keys.add(key)
        task_trials[key[:2]] += 1
        state = (
            int(episode["init_state_id"]),
            int(episode["num_init_states"]),
            int(episode["seed"]),
        )
        if state[0] != key[2] % state[1] or state[2] != SEED:
            raise RuntimeError(f"{label} episode {key} violates the initial-state rule")
        if (
            int(episode["initialization_steps"]) != INITIALIZATION_STEPS
            or int(episode["max_policy_steps"]) != MAX_POLICY_STEPS
            or int(episode["policy_steps"]) != int(episode["steps"])
        ):
            raise RuntimeError(f"{label} episode {key} violates the rollout horizon")
        if reference is None:
            initial_states[key] = state
        elif initial_states.get(key) != state:
            raise RuntimeError(f"{label} episode {key} does not match native initial state and seed")

    if set(episode["benchmark"] for episode in episodes) != set(BENCHMARKS):
        raise RuntimeError(f"{label} does not contain all LIBERO-plus suites")
    invalid_trials = {key: count for key, count in task_trials.items() if count != EPISODES_PER_TASK}
    if invalid_trials or len(task_trials) != sum(task_counts.values()):
        raise RuntimeError(f"{label} does not contain exactly {EPISODES_PER_TASK} trials per task")
    if reference is not None and episode_keys != set(initial_states):
        raise RuntimeError(f"{label} episode catalog differs from native")

    recomputed = summarize(episodes)
    wall_time = shard_wall_time(protocol_root, len(commands))
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
            "summary_json": str(summary_path.resolve()),
            "commands_json": str(commands_path.resolve()),
            "launcher_command": manifest.get("launcher_command"),
            "task_counts": task_counts,
            "episodes_per_task": EPISODES_PER_TASK,
            "total_episodes": expected_episodes,
            "seed": SEED,
            "action_infer_steps": ACTION_INFER_STEPS,
            "official_evaluation": official,
            "wall_clock": wall_time,
            "summary": recomputed,
        },
    }


def csv_rows(weights):
    for label, result in weights.items():
        overall = result["summary"]["overall"]
        yield {"weight": label, "scope": "overall", "suite": "", "task_id": "", "task_name": "", "category": "", "difficulty_level": "", **overall}
        for suite, item in result["summary"]["suites"].items():
            yield {"weight": label, "scope": "suite", "suite": suite, "task_id": "", "task_name": "", "category": "", "difficulty_level": "", **item}
        for task_key, item in result["summary"]["tasks"].items():
            suite, task_id = task_key.rsplit("/", 1)
            row = {
                "weight": label,
                "scope": "task",
                "suite": suite,
                "task_id": task_id,
                "task_name": item["task_name"],
                "category": item.get("category"),
                "difficulty_level": item.get("difficulty_level"),
            }
            row.update({key: value for key, value in item.items() if key not in {"task_name", "category", "difficulty_level"}})
            yield row


def atomic_csv_dump(weights, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames = [
        "weight", "scope", "suite", "task_id", "task_name", "category", "difficulty_level",
        "episodes", "successes", "failures", "success_rate", "average_policy_steps",
        "average_success_policy_steps", "rollout_elapsed_seconds",
        "average_rollout_elapsed_seconds", "failure_reasons",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows(weights):
            row["failure_reasons"] = json.dumps(row["failure_reasons"], sort_keys=True)
            writer.writerow(row)
    os.replace(temporary, path)


def main():
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else results_root / "comparison_summary.json"
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else results_root / "comparison_summary.csv"
    if len(args.weight) != 4:
        raise ValueError("Exactly four ordered --weight entries are required")

    weights = {}
    reference = None
    for label, directory, adapter in args.weight:
        if label in weights:
            raise ValueError(f"Duplicate weight label: {label}")
        result_dir = Path(directory)
        if not result_dir.is_absolute():
            result_dir = results_root / result_dir
        validated = validate_weight(label, result_dir.resolve(), adapter, reference)
        if reference is None:
            reference = validated["reference"]
        weights[label] = validated["output"]

    payload = {
        "generated_at": utc_now(),
        "results_root": str(results_root),
        "verification": {
            "weights_verified": list(weights),
            "all_suites_and_tasks_complete": True,
            "same_episode_catalog_seed_and_initial_states": True,
            "official_protocol_and_implementation_match": True,
            "episodes_per_task": EPISODES_PER_TASK,
            "total_tasks_per_weight": sum(reference["task_counts"].values()),
            "total_episodes_per_weight": sum(reference["task_counts"].values()) * EPISODES_PER_TASK,
        },
        "weights": weights,
    }
    atomic_json_dump(payload, output_json)
    atomic_csv_dump(weights, output_csv)
    print(f"[verified] json={output_json} csv={output_csv}")


if __name__ == "__main__":
    main()
