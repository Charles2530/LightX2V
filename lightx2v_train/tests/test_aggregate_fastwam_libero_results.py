import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import aggregate_fastwam_libero_results as aggregate


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_weight(results_root, label, adapter):
    protocol_root = results_root / label / "official_protocol"
    official = {
        "initialization_steps": 5,
        "initialization_action": [0.0] * 7,
        "max_policy_steps": 600,
        "metric_script_sha256": "metric-hash",
    }
    implementation = {"evaluator.py": {"sha256": "implementation-hash"}}
    task_counts = {benchmark: 1 for benchmark in aggregate.BENCHMARKS}
    episodes = []
    for benchmark in aggregate.BENCHMARKS:
        for episode_index in range(50):
            success = episode_index != 49
            episodes.append(
                {
                    "benchmark": benchmark,
                    "task_id": 0,
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
                    "task_name": f"{benchmark}_task",
                    "category": "test",
                    "difficulty_level": 1,
                }
            )
    commands = [{"shard": benchmark} for benchmark in aggregate.BENCHMARKS]
    manifest = {
        "adapter": str(adapter.resolve()),
        "launcher_command": "synthetic launcher",
        "task_counts": task_counts,
        "episodes_per_task": 50,
        "seed": 0,
        "expected_action_infer_steps": 1,
        "devices": [1, 2, 3, 4, 5, 6, 7],
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
        "commands": commands,
    }
    summary = {
        "adapter": str(adapter.resolve()),
        "action_infer_steps": 1,
        "seed": 0,
        "max_policy_steps": 600,
        "evaluation_protocol": "50_episodes_per_task",
        "finished_at": "2026-01-01T00:01:00Z",
        "task_counts": task_counts,
        "protocols_verified": len(commands),
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
        "episodes": episodes,
    }
    write_json(protocol_root / "commands.json", manifest)
    write_json(protocol_root / "summary.json", summary)
    for index, benchmark in enumerate(aggregate.BENCHMARKS):
        write_json(
            protocol_root / "shards" / f"{benchmark}.json",
            {
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": f"2026-01-01T00:00:{index + 1:02d}Z",
            },
        )


def test_verifies_four_weights_and_writes_comparison(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    weights = []
    for label in ("native", "old", "lora_only", "joint"):
        adapter = tmp_path / f"{label}.pt"
        make_weight(results_root, label, adapter)
        weights.extend(["--weight", label, label, str(adapter)])

    monkeypatch.setattr(
        sys,
        "argv",
        ["aggregate_fastwam_libero_results.py", "--results-root", str(results_root), *weights],
    )
    aggregate.main()

    comparison = json.loads((results_root / "comparison_summary.json").read_text())
    assert comparison["verification"]["weights_verified"] == [
        "native",
        "old",
        "lora_only",
        "joint",
    ]
    assert comparison["verification"]["total_tasks_per_weight"] == 4
    assert comparison["verification"]["total_episodes_per_weight"] == 200
    assert comparison["weights"]["native"]["summary"]["overall"]["success_rate"] == 0.98
    assert comparison["weights"]["joint"]["wall_clock"]["elapsed_seconds"] == 4.0
    assert len((results_root / "comparison_summary.csv").read_text().splitlines()) == 37
