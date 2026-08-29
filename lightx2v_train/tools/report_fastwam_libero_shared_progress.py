"""Report and audit partial shared-policy LIBERO-plus evaluation progress."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import aggregate_fastwam_libero_results as common


DEFAULT_WEIGHTS = (
    ("native", "native"),
    ("old_success_baseline_30k", "old_success_baseline_30k"),
    ("lora_only_30k", "lora_only_30k"),
    ("joint_30k", "joint_30k"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--protocol-directory",
        default="official_protocol_shared_policy",
    )
    parser.add_argument(
        "--weight",
        action="append",
        nargs=2,
        metavar=("LABEL", "RESULT_DIRECTORY"),
        help="Repeat to override the four canonical weight directories",
    )
    parser.add_argument("--output", help="Optionally write the report atomically as JSON")
    return parser.parse_args()


def stable_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def common_run_signature(signature):
    signature = dict(signature)
    signature.pop("benchmarks", None)
    signature.pop("task_ids", None)
    return signature


def load_manifest(protocol_root):
    path = protocol_root / "commands.json"
    if not path.is_file():
        return None
    return common.load_json(path)


def expected_reference(results_root, protocol_directory, weights):
    for _, directory in weights:
        result_dir = Path(directory)
        if not result_dir.is_absolute():
            result_dir = results_root / result_dir
        manifest = load_manifest(result_dir / protocol_directory)
        if manifest is not None:
            return {
                "task_counts": manifest.get("task_counts", {}),
                "episodes_per_task": int(manifest.get("episodes_per_task", common.EPISODES_PER_TASK)),
                "seed": int(manifest.get("seed", common.SEED)),
                "action_infer_steps": int(
                    manifest.get("expected_action_infer_steps", common.ACTION_INFER_STEPS)
                ),
                "actions_per_plan": int(manifest.get("expected_actions_per_plan", 10)),
                "official_evaluation": manifest.get("official_evaluation"),
                "implementation": manifest.get("fastwam_evaluation_implementation"),
            }
    return {
        "task_counts": {},
        "episodes_per_task": common.EPISODES_PER_TASK,
        "seed": common.SEED,
        "action_infer_steps": common.ACTION_INFER_STEPS,
        "actions_per_plan": 10,
        "official_evaluation": None,
        "implementation": None,
    }


def partial_scores(episodes):
    suites = common.grouped_scores(episodes, lambda item: item["benchmark"])
    return {
        "overall": common.score(episodes),
        "suites": suites,
    }


def audit_episode(episode, reference, violations):
    episode_index = int(episode["episode_index"])
    num_init_states = int(episode["num_init_states"])
    official = reference.get("official_evaluation") or {}
    if int(episode["seed"]) != reference["seed"]:
        violations["seed"] += 1
    if episode_index < 0 or episode_index >= reference["episodes_per_task"]:
        violations["episode_index"] += 1
    if num_init_states <= 0 or int(episode["init_state_id"]) != episode_index % num_init_states:
        violations["initial_state_rule"] += 1
    if official and int(episode["initialization_steps"]) != int(official["initialization_steps"]):
        violations["initialization_steps"] += 1
    if official and int(episode["max_policy_steps"]) != int(official["max_policy_steps"]):
        violations["max_policy_steps"] += 1
    if int(episode["policy_steps"]) != int(episode["steps"]):
        violations["policy_steps"] += 1
    if int(episode["total_env_steps"]) != int(episode["policy_steps"]) + int(
        episode["initialization_steps"]
    ):
        violations["total_env_steps"] += 1
    policy_steps = int(episode["policy_steps"])
    max_policy_steps = int(episode["max_policy_steps"])
    if not 1 <= policy_steps <= max_policy_steps:
        violations["policy_step_horizon"] += 1
    if bool(episode["success"]):
        if episode.get("failure_reason") is not None:
            violations["success_failure_reason"] += 1
    else:
        if episode.get("failure_reason") != "max_steps_exceeded":
            violations["failure_reason"] += 1
        if policy_steps != max_policy_steps:
            violations["failure_not_at_horizon"] += 1


def audit_protocol(payload, manifest, reference, violations):
    protocol = payload.get("protocol", {})
    if int(protocol.get("action_infer_steps", -1)) != reference["action_infer_steps"]:
        violations["protocol_action_infer_steps"] += 1
    if int(protocol.get("actions_per_plan", -1)) != reference["actions_per_plan"]:
        violations["protocol_actions_per_plan"] += 1
    if int(protocol.get("seed", -1)) != reference["seed"]:
        violations["protocol_seed"] += 1
    if reference.get("official_evaluation") is not None and protocol.get(
        "official_evaluation"
    ) != reference["official_evaluation"]:
        violations["official_evaluation"] += 1
    if reference.get("implementation") is not None and protocol.get(
        "fastwam_evaluation_implementation"
    ) != reference["implementation"]:
        violations["implementation"] += 1
    if manifest is not None and payload.get("run_signature", {}).get("adapter") != manifest.get(
        "adapter"
    ):
        violations["adapter"] += 1


def report_weight(label, result_dir, protocol_directory, reference):
    protocol_root = result_dir / protocol_directory
    manifest = load_manifest(protocol_root)
    shard_paths = sorted((protocol_root / "shards").glob("*.json"))
    episodes = []
    invalid_json = []
    complete_shards = 0
    violations = Counter()
    signature_counts = Counter()

    for path in shard_paths:
        try:
            payload = common.load_json(path)
        except (OSError, ValueError) as error:
            invalid_json.append({"path": str(path.resolve()), "error": str(error)})
            continue
        shard_episodes = payload.get("episodes", [])
        episodes.extend(shard_episodes)
        if payload.get("finished_at") and len(shard_episodes) == reference["episodes_per_task"]:
            complete_shards += 1
        signature_counts[stable_digest(common_run_signature(payload.get("run_signature", {})))] += 1
        audit_protocol(payload, manifest, reference, violations)
        for episode in shard_episodes:
            audit_episode(episode, reference, violations)

    episode_keys = [
        (item["benchmark"], int(item["task_id"]), int(item["episode_index"]))
        for item in episodes
    ]
    duplicate_episodes = len(episode_keys) - len(set(episode_keys))
    if duplicate_episodes:
        violations["duplicate_episode_keys"] += duplicate_episodes

    task_trials = Counter(key[:2] for key in episode_keys)
    overfilled_tasks = sum(
        count > reference["episodes_per_task"] for count in task_trials.values()
    )
    if overfilled_tasks:
        violations["overfilled_tasks"] += overfilled_tasks
    trial_histogram = Counter(task_trials.values())
    expected_tasks = sum(reference["task_counts"].values())
    expected_episodes = expected_tasks * reference["episodes_per_task"]

    return {
        "label": label,
        "result_directory": str(result_dir.resolve()),
        "protocol_root": str(protocol_root.resolve()),
        "manifest_present": manifest is not None,
        "adapter": manifest.get("adapter") if manifest is not None else None,
        "expected_tasks": expected_tasks,
        "expected_episodes": expected_episodes,
        "shard_files": len(shard_paths),
        "complete_shards": complete_shards,
        "tasks_seen": len(task_trials),
        "complete_tasks": sum(
            count == reference["episodes_per_task"] for count in task_trials.values()
        ),
        "trial_count_histogram": {
            str(count): tasks for count, tasks in sorted(trial_histogram.items())
        },
        "episodes_recorded": len(episodes),
        "progress_fraction": len(episodes) / expected_episodes if expected_episodes else 0.0,
        "partial_summary": partial_scores(episodes),
        "invalid_json_files": invalid_json,
        "run_signature_variants": dict(sorted(signature_counts.items())),
        "partial_invariant_violations": dict(sorted(violations.items())),
        "partial_invariants_valid": not invalid_json and not violations,
    }


def build_report(results_root, protocol_directory, weights):
    reference = expected_reference(results_root, protocol_directory, weights)
    reports = {}
    for label, directory in weights:
        if label in reports:
            raise ValueError(f"Duplicate weight label: {label}")
        result_dir = Path(directory)
        if not result_dir.is_absolute():
            result_dir = results_root / result_dir
        reports[label] = report_weight(
            label,
            result_dir.resolve(),
            protocol_directory,
            reference,
        )
    expected_per_weight = sum(reference["task_counts"].values()) * reference["episodes_per_task"]
    recorded = sum(item["episodes_recorded"] for item in reports.values())
    expected = expected_per_weight * len(reports)
    return {
        "generated_at": common.utc_now(),
        "results_root": str(results_root.resolve()),
        "protocol_directory": protocol_directory,
        "ordered_weights": list(reports),
        "reference": reference,
        "overall_progress": {
            "episodes_recorded": recorded,
            "episodes_expected": expected,
            "progress_fraction": recorded / expected if expected else 0.0,
        },
        "all_partial_invariants_valid": all(
            item["partial_invariants_valid"] for item in reports.values()
        ),
        "weights": reports,
    }


def main():
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    weights = tuple(args.weight) if args.weight else DEFAULT_WEIGHTS
    report = build_report(results_root, args.protocol_directory, weights)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        common.atomic_json_dump(report, output)
        print(f"[progress] output={output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
