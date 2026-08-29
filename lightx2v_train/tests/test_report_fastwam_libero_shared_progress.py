import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


import report_fastwam_libero_shared_progress as progress


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reports_partial_four_weight_progress(tmp_path):
    results_root = tmp_path / "results"
    protocol_root = results_root / "native" / "protocol"
    official = {
        "initialization_steps": 5,
        "max_policy_steps": 600,
    }
    implementation = {"shared.py": {"sha256": "implementation-hash"}}
    manifest = {
        "adapter": "/models/native.pt",
        "task_counts": {"libero_spatial": 1},
        "episodes_per_task": 50,
        "seed": 0,
        "expected_action_infer_steps": 1,
        "expected_actions_per_plan": 10,
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
    }
    write_json(protocol_root / "commands.json", manifest)
    episodes = []
    for episode_index, success in ((0, True), (1, False)):
        steps = 100 if success else 600
        episodes.append(
            {
                "benchmark": "libero_spatial",
                "task_id": 0,
                "episode_index": episode_index,
                "init_state_id": episode_index,
                "num_init_states": 50,
                "seed": 0,
                "initialization_steps": 5,
                "max_policy_steps": 600,
                "steps": steps,
                "policy_steps": steps,
                "total_env_steps": steps + 5,
                "success": success,
                "failure_reason": None if success else "max_steps_exceeded",
                "elapsed_seconds": 1.0,
            }
        )
    write_json(
        protocol_root / "shards" / "libero_spatial-0.json",
        {
            "run_signature": {
                "adapter": manifest["adapter"],
                "benchmarks": ["libero_spatial"],
                "task_ids": {"libero_spatial": [0]},
                "seed": 0,
            },
            "protocol": {
                "action_infer_steps": 1,
                "actions_per_plan": 10,
                "seed": 0,
                "official_evaluation": official,
                "fastwam_evaluation_implementation": implementation,
            },
            "episodes": episodes,
        },
    )

    report = progress.build_report(
        results_root,
        "protocol",
        progress.DEFAULT_WEIGHTS,
    )

    assert report["ordered_weights"] == [
        "native",
        "old_success_baseline_30k",
        "lora_only_30k",
        "joint_30k",
    ]
    assert report["overall_progress"] == {
        "episodes_recorded": 2,
        "episodes_expected": 200,
        "progress_fraction": 0.01,
    }
    native = report["weights"]["native"]
    assert native["episodes_recorded"] == 2
    assert native["partial_summary"]["overall"]["success_rate"] == 0.5
    assert native["partial_summary"]["overall"]["failure_reasons"] == {
        "max_steps_exceeded": 1
    }
    assert native["trial_count_histogram"] == {"2": 1}
    assert native["partial_invariants_valid"] is True
    assert len(native["run_signature_variants"]) == 1
    assert report["weights"]["joint_30k"]["episodes_recorded"] == 0
    assert report["all_partial_invariants_valid"] is True


def test_detects_duplicate_and_seed_violation(tmp_path):
    results_root = tmp_path / "results"
    protocol_root = results_root / "native" / "protocol"
    manifest = {
        "adapter": "/models/native.pt",
        "task_counts": {"libero_spatial": 1},
        "episodes_per_task": 50,
        "seed": 0,
        "expected_action_infer_steps": 1,
        "expected_actions_per_plan": 10,
        "official_evaluation": {"initialization_steps": 5, "max_policy_steps": 600},
        "fastwam_evaluation_implementation": {},
    }
    write_json(protocol_root / "commands.json", manifest)
    episode = {
        "benchmark": "libero_spatial",
        "task_id": 0,
        "episode_index": 0,
        "init_state_id": 0,
        "num_init_states": 50,
        "seed": 1,
        "initialization_steps": 5,
        "max_policy_steps": 600,
        "steps": 100,
        "policy_steps": 100,
        "total_env_steps": 105,
        "success": True,
        "failure_reason": None,
        "elapsed_seconds": 1.0,
    }
    for task_id in (0, 1):
        write_json(
            protocol_root / "shards" / f"shard-{task_id}.json",
            {
                "run_signature": {"adapter": manifest["adapter"], "task_ids": [task_id]},
                "protocol": {
                    "action_infer_steps": 1,
                    "actions_per_plan": 10,
                    "seed": 0,
                    "official_evaluation": manifest["official_evaluation"],
                    "fastwam_evaluation_implementation": {},
                },
                "episodes": [episode],
            },
        )

    report = progress.build_report(
        results_root,
        "protocol",
        (("native", "native"),),
    )

    native = report["weights"]["native"]
    assert native["partial_invariant_violations"]["duplicate_episode_keys"] == 1
    assert native["partial_invariant_violations"]["seed"] == 2
    assert native["partial_invariants_valid"] is False
    assert report["all_partial_invariants_valid"] is False


def test_detects_failed_episode_before_horizon():
    violations = progress.Counter()
    episode = {
        "episode_index": 0,
        "num_init_states": 50,
        "init_state_id": 0,
        "seed": 0,
        "initialization_steps": 5,
        "max_policy_steps": 600,
        "steps": 100,
        "policy_steps": 100,
        "total_env_steps": 105,
        "success": False,
        "failure_reason": "max_steps_exceeded",
    }
    reference = {
        "seed": 0,
        "episodes_per_task": 50,
        "official_evaluation": {"initialization_steps": 5, "max_policy_steps": 600},
    }

    progress.audit_episode(episode, reference, violations)

    assert violations == {"failure_not_at_horizon": 1}
