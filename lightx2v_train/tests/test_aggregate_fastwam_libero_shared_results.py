import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import aggregate_fastwam_libero_shared_results as aggregate


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_weight(results_root, label, adapter):
    protocol_root = results_root / label / "official_protocol_shared_policy"
    official = {
        "initialization_steps": 5,
        "initialization_action": [0.0] * 7,
        "max_policy_steps": 600,
        "metric_script_sha256": "metric-hash",
    }
    implementation = {"shared.py": {"sha256": "implementation-hash"}}
    task_counts = {benchmark: 1 for benchmark in aggregate.common.BENCHMARKS}
    task_counts["libero_spatial"] = 4
    tasks = []
    for benchmark, count in task_counts.items():
        tasks.extend((benchmark, task_id) for task_id in range(count))

    episodes = []
    for benchmark, task_id in tasks:
        for episode_index in range(50):
            success = episode_index != 49
            episodes.append(
                {
                    "benchmark": benchmark,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "init_state_id": episode_index,
                    "num_init_states": 50,
                    "seed": 0,
                    "initialization_steps": 5,
                    "max_policy_steps": 600,
                    "steps": 100 if success else 600,
                    "policy_steps": 100 if success else 600,
                    "success": success,
                    "failure_reason": None if success else "max_steps_exceeded",
                    "elapsed_seconds": 1.0,
                    "task_name": f"{benchmark}_{task_id}",
                    "category": "test",
                    "difficulty_level": 1,
                }
            )

    commands = [{"command": f"server {device}"} for device in range(1, 8)]
    manifest = {
        "adapter": str(adapter.resolve()),
        "launcher_command": "synthetic launcher",
        "evaluation_mode": aggregate.EVALUATION_MODE,
        "task_counts": task_counts,
        "episodes_per_task": 50,
        "expected_shards": len(tasks),
        "seed": 0,
        "expected_action_infer_steps": 1,
        "expected_actions_per_plan": 10,
        "prompt_cache_mode": "lazy-lru",
        "text_encoder_released": False,
        "devices": list(range(1, 8)),
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
        "commands": commands,
    }
    summary = {
        "adapter": str(adapter.resolve()),
        "evaluation_mode": aggregate.EVALUATION_MODE,
        "action_infer_steps": 1,
        "actions_per_plan": 10,
        "seed": 0,
        "max_policy_steps": 600,
        "evaluation_protocol": "50_episodes_per_task",
        "finished_at": "2026-01-01T00:01:00Z",
        "task_counts": task_counts,
        "protocols_verified": len(tasks),
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
        "episodes": episodes,
    }
    write_json(protocol_root / "commands.json", manifest)
    write_json(protocol_root / "summary.json", summary)
    for index, (benchmark, task_id) in enumerate(tasks):
        write_json(
            protocol_root / "shards" / f"{benchmark}-{task_id}.json",
            {
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": f"2026-01-01T00:00:{index + 1:02d}Z",
                "protocol": {
                    "action_infer_steps": 1,
                    "actions_per_plan": 10,
                    "seed": 0,
                    "prompt_cache_mode": "lazy-lru",
                    "text_encoder_released": False,
                    "physical_cuda_device": index + 1,
                    "official_evaluation": official,
                    "fastwam_evaluation_implementation": implementation,
                },
            },
        )


def test_verifies_four_shared_policy_weights(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    weights = []
    for label in ("native", "old", "lora_only", "joint"):
        adapter = tmp_path / f"{label}.pt"
        make_weight(results_root, label, adapter)
        weights.extend(["--weight", label, label, str(adapter)])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_fastwam_libero_shared_results.py",
            "--results-root",
            str(results_root),
            *weights,
        ],
    )
    aggregate.main()

    comparison = json.loads((results_root / "comparison_summary.json").read_text())
    assert comparison["verification"]["weights_verified"] == [
        "native",
        "old",
        "lora_only",
        "joint",
    ]
    assert comparison["verification"]["total_tasks_per_weight"] == 7
    assert comparison["verification"]["total_episodes_per_weight"] == 350
    assert comparison["weights"]["native"]["summary"]["overall"]["success_rate"] == 0.98
    assert comparison["weights"]["joint"]["wall_clock"]["elapsed_seconds"] == 7.0
    assert len((results_root / "comparison_summary.csv").read_text().splitlines()) == 49
