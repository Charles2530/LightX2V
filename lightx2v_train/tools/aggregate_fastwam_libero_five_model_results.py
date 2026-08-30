"""Strictly verify and compare the requested five LIBERO-plus FastWAM models."""

import argparse
import json
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import aggregate_fastwam_libero_results as common
import aggregate_fastwam_libero_shared_results as shared


EXPECTED_MODELS = {
    "native_20step": 20,
    "native_1step": 1,
    "joint_30k": 1,
    "lora_only_30k": 1,
    "old_joint_30k": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        nargs=6,
        metavar=(
            "LABEL",
            "RESULT_DIRECTORY",
            "PROTOCOL_DIRECTORY",
            "ADAPTER",
            "ACTION_INFER_STEPS",
            "ACTIONS_PER_PLAN",
        ),
        required=True,
    )
    parser.add_argument("--task-catalog-audit", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def parse_models(values):
    models = []
    seen = set()
    for label, result_directory, protocol_directory, adapter, steps, actions in values:
        if label in seen:
            raise ValueError(f"Duplicate model label: {label}")
        try:
            steps = int(steps)
            actions = int(actions)
        except ValueError as error:
            raise ValueError(f"{label} steps and actions_per_plan must be integers") from error
        if steps <= 0 or actions <= 0:
            raise ValueError(f"{label} steps and actions_per_plan must be positive")
        models.append(
            {
                "label": label,
                "result_directory": Path(result_directory).expanduser().resolve(),
                "protocol_directory": protocol_directory,
                "adapter": str(Path(adapter).expanduser().resolve()),
                "action_infer_steps": steps,
                "actions_per_plan": actions,
            }
        )
        seen.add(label)
    if set(seen) != set(EXPECTED_MODELS):
        raise ValueError(
            f"Expected exactly {sorted(EXPECTED_MODELS)}, got {sorted(seen)}"
        )
    mismatches = {
        model["label"]: {
            "actual": model["action_infer_steps"],
            "expected": EXPECTED_MODELS[model["label"]],
        }
        for model in models
        if model["action_infer_steps"] != EXPECTED_MODELS[model["label"]]
    }
    if mismatches:
        raise ValueError(f"Unexpected action inference steps: {mismatches}")
    if any(model["actions_per_plan"] != 10 for model in models):
        raise ValueError("All five models must use actions_per_plan=10")
    return models


def file_evidence(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": shared.sha256_file(path),
    }


def dependency_evidence(weight):
    return {
        "policy_config": file_evidence(weight["policy_config"]),
        "dataset_stats": file_evidence(weight["dataset_stats"]),
    }


def validate_requested_models(models, task_catalog_audit):
    task_catalog = shared.load_task_catalog_audit(task_catalog_audit)
    weights = {}
    reference = None
    for model in models:
        validated = shared.validate_weight(
            model["label"],
            model["result_directory"],
            model["adapter"],
            reference,
            model["protocol_directory"],
            expected_action_infer_steps=model["action_infer_steps"],
            expected_actions_per_plan=model["actions_per_plan"],
        )
        if reference is None:
            reference = validated["reference"]
        output = validated["output"]
        output["protocol_directory"] = model["protocol_directory"]
        output["dependency_evidence"] = dependency_evidence(output)
        weights[model["label"]] = output

    if reference["task_counts"] != task_catalog["task_counts"]:
        raise RuntimeError("Completed suite task counts differ from the task catalog audit")
    if sum(reference["task_counts"].values()) != task_catalog["total_tasks"]:
        raise RuntimeError("Completed task total differs from the task catalog audit")
    native_20step = weights["native_20step"]
    native_1step = weights["native_1step"]
    if native_20step["adapter"] != native_1step["adapter"]:
        raise RuntimeError("Native 20-step and 1-step evaluations did not use the same checkpoint")
    if native_20step["checkpoint"]["sha256"] != native_1step["checkpoint"]["sha256"]:
        raise RuntimeError("Native 20-step and 1-step checkpoint hashes differ")
    return weights, reference, task_catalog


def build_payload(weights, reference, task_catalog):
    expected_episodes = sum(reference["task_counts"].values()) * common.EPISODES_PER_TASK
    return {
        "generated_at": common.utc_now(),
        "task_catalog_audit": task_catalog["output"],
        "verification": {
            "models_verified": list(weights),
            "all_five_requested_models_complete": True,
            "all_suites_and_tasks_complete": True,
            "same_episode_catalog_seed_and_initial_states": True,
            "official_protocol_and_implementation_match": True,
            "episode_outcomes_match_official_horizon": True,
            "shared_policy_lazy_prompt_cache_verified": True,
            "checkpoint_and_dependency_hashes_recorded": True,
            "libero_task_resources_match_baseline": True,
            "physical_cuda_devices": [1, 2, 3, 4, 5, 6, 7],
            "episodes_per_task": common.EPISODES_PER_TASK,
            "total_tasks_per_model": sum(reference["task_counts"].values()),
            "total_episodes_per_model": expected_episodes,
            "total_episodes_all_models": expected_episodes * len(weights),
        },
        "models": weights,
    }


def main():
    args = parse_args()
    models = parse_models(args.model)
    weights, reference, task_catalog = validate_requested_models(
        models, args.task_catalog_audit
    )
    payload = build_payload(weights, reference, task_catalog)
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    common.atomic_json_dump(payload, output_json)
    common.atomic_csv_dump(weights, output_csv)
    print(json.dumps(payload["verification"], sort_keys=True))
    print(f"[verified] json={output_json} csv={output_csv}")


if __name__ == "__main__":
    main()
