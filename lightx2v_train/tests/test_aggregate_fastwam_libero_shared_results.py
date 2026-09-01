import json
import sys
from pathlib import Path

import pytest


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import aggregate_fastwam_libero_shared_results as aggregate
import aggregate_fastwam_libero_five_model_results as five_model


def log_manifest(path, command):
    return {
        "commands": [
            {
                "physical_cuda_device": 1,
                "log": str(path),
                "command": command,
            }
        ]
    }


def test_rejects_log_without_one_step_configuration(tmp_path):
    command = "python eval_fastwam_libero_shared_policy.py --seed 0"
    log = tmp_path / "server.log"
    log.write_text(f"[2026-01-01T00:00:00Z] {command}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing"):
        aggregate.validate_server_logs(log_manifest(log, command))


def test_rejects_log_with_error_marker(tmp_path):
    command = (
        "python eval_fastwam_libero_shared_policy.py "
        "--expected-action-infer-steps 1 --expected-actions-per-plan 10 --seed 0"
    )
    log = tmp_path / "server.log"
    log.write_text(
        f"[2026-01-01T00:00:00Z] {command}\nTraceback (most recent call last):\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="error markers"):
        aggregate.validate_server_logs(log_manifest(log, command))


def test_latest_log_segment_ignores_but_records_historical_failures(tmp_path):
    old_command = (
        "python eval_fastwam_libero_shared_policy.py "
        "--expected-action-infer-steps 1 --expected-actions-per-plan 10 --seed 0 --env-workers 8"
    )
    command = old_command.replace("--env-workers 8", "--env-workers 1")
    log = tmp_path / "server.log"
    log.write_text(
        f"[2026-01-01T00:00:00Z] {old_command}\n"
        "Traceback (most recent call last):\n"
        f"[2026-01-02T00:00:00Z] {command}\n"
        "current rollout output\n",
        encoding="utf-8",
    )

    evidence = aggregate.validate_server_logs(log_manifest(log, command))[0]
    assert evidence["launch_count"] == 2
    assert evidence["historical_launch_count"] == 1
    assert evidence["historical_error_markers"] == 1
    assert evidence["current_launch_line_number"] == 3
    assert evidence["error_markers"] == 0


def test_rejects_checkpoint_changed_after_snapshot(tmp_path):
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"current")
    expected = {
        "path": str(checkpoint),
        "size_bytes": len(b"frozen!"),
        "sha256": aggregate.hashlib.sha256(b"frozen!").hexdigest(),
    }

    with pytest.raises(RuntimeError, match="SHA256 differs"):
        aggregate.checkpoint_evidence(checkpoint, expected)


def test_directory_content_digest_matches_sha256sum_records(tmp_path):
    root = tmp_path / "libero" / "libero"
    resources = root / "bddl_files"
    (resources / "suite").mkdir(parents=True)
    (resources / "suite" / "one.bddl").write_text("one", encoding="utf-8")
    (resources / "two.bddl").write_text("two", encoding="utf-8")
    expected = aggregate.hashlib.sha256()
    for relative in ("bddl_files/suite/one.bddl", "bddl_files/two.bddl"):
        item = root / relative
        expected.update(f"{aggregate.sha256_file(item)}  {relative}\n".encode("utf-8"))

    assert aggregate.directory_content_digest(resources, root) == (2, expected.hexdigest())


@pytest.mark.parametrize(
    ("episode", "message"),
    (
        (
            {
                "initialization_steps": 5,
                "policy_steps": 100,
                "max_policy_steps": 600,
                "total_env_steps": 105,
                "success": False,
                "failure_reason": "max_steps_exceeded",
            },
            "did not exhaust",
        ),
        (
            {
                "initialization_steps": 5,
                "policy_steps": 100,
                "max_policy_steps": 600,
                "total_env_steps": 104,
                "success": True,
                "failure_reason": None,
            },
            "inconsistent total",
        ),
    ),
)
def test_rejects_invalid_episode_outcome(episode, message):
    with pytest.raises(RuntimeError, match=message):
        aggregate.validate_episode_outcome("native", ("suite", 0, 0), episode)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_weight(
    results_root,
    label,
    adapter,
    render_backend=None,
    action_infer_steps=1,
    protocol_directory="official_protocol_shared_policy",
):
    protocol_root = results_root / label / protocol_directory
    metric_path = results_root / "metric.py"
    metric_sha256 = aggregate.sha256_file(metric_path) if metric_path.is_file() else "metric-hash"
    official = {
        "initialization_steps": 5,
        "initialization_action": [0.0] * 7,
        "max_policy_steps": 600,
        "metric_script_sha256": metric_sha256,
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
                    "total_env_steps": 105 if success else 605,
                    "success": success,
                    "failure_reason": None if success else "max_steps_exceeded",
                    "elapsed_seconds": 1.0,
                    "task_name": f"{benchmark}_{task_id}",
                    "category": "test",
                    "difficulty_level": 1,
                }
            )

    adapter.write_bytes(label.encode("utf-8"))
    commands = []
    for device in range(1, 8):
        log = protocol_root / f"server-cuda-{device}.log"
        command = (
            "python eval_fastwam_libero_shared_policy.py "
            f"--expected-action-infer-steps {action_infer_steps} "
            "--expected-actions-per-plan 10 --seed 0"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"\n[2026-01-01T00:00:00Z] {command}\n", encoding="utf-8")
        commands.append(
            {
                "physical_cuda_device": device,
                "cuda_visible_devices": str(device),
                "mujoco_egl_device_id": str(device),
                "log": str(log.resolve()),
                "command": command,
            }
        )
    manifest = {
        "adapter": str(adapter.resolve()),
        "model_path": str((results_root / "model").resolve()),
        "dataset_stats": str((results_root / "stats.json").resolve()),
        "policy_config": str((results_root / "policy.json").resolve()),
        "libero_root": str((results_root / "libero").resolve()),
        "launcher_command": "synthetic launcher",
        "evaluation_mode": aggregate.EVALUATION_MODE,
        "task_counts": task_counts,
        "episodes_per_task": 50,
        "expected_shards": len(tasks),
        "seed": 0,
        "expected_action_infer_steps": action_infer_steps,
        "expected_actions_per_plan": 10,
        "prompt_cache_mode": "lazy-lru",
        "text_encoder_released": False,
        "devices": list(range(1, 8)),
        "render_backend": render_backend,
        "official_evaluation": official,
        "fastwam_evaluation_implementation": implementation,
        "commands": commands,
    }
    summary = {
        "adapter": str(adapter.resolve()),
        "evaluation_mode": aggregate.EVALUATION_MODE,
        "action_infer_steps": action_infer_steps,
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
                    "action_infer_steps": action_infer_steps,
                    "actions_per_plan": 10,
                    "seed": 0,
                    "prompt_cache_mode": "lazy-lru",
                    "text_encoder_released": False,
                    "physical_cuda_device": index + 1,
                    "render_environment": (
                        {
                            "mujoco_gl": "egl",
                            "pyopengl_platform": "egl",
                            "egl_platform": "surfaceless",
                            "cuda_visible_devices": str(index + 1),
                            "mujoco_egl_device_id": str(index + 1),
                            "nvidia_egl_root": render_backend["root"],
                            "__egl_vendor_library_filenames": render_backend["files"][
                                "egl_vendor_json"
                            ]["path"],
                        }
                        if render_backend is not None
                        else None
                    ),
                    "official_evaluation": official,
                    "fastwam_evaluation_implementation": implementation,
                },
            },
        )


def test_verifies_four_shared_policy_weights(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    (results_root / "model").mkdir(parents=True)
    (results_root / "stats.json").write_text("stats", encoding="utf-8")
    (results_root / "policy.json").write_text("policy", encoding="utf-8")
    evaluation_root = results_root / "evaluation"
    evaluation_root.mkdir()
    (evaluation_root / "shared.py").write_text("evaluation", encoding="utf-8")
    official_metric = results_root / "metric.py"
    official_config = results_root / "eval.yaml"
    official_metric.write_text("metric", encoding="utf-8")
    official_config.write_text("max_steps: 600", encoding="utf-8")
    render_root = results_root / "nvidia-egl"
    render_files = {}
    for name in (
        "egl_vendor_json",
        "libegl_nvidia",
        "libnvidia_eglcore",
        "libnvidia_glsi",
    ):
        path = render_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        render_files[name] = {
            "path": str(path.resolve()),
            "sha256": aggregate.sha256_file(path),
        }
    render_backend = {
        "name": "nvidia-egl",
        "root": str(render_root.resolve()),
        "library_dir": str(render_root.resolve()),
        "physical_device_rule": "MUJOCO_EGL_DEVICE_ID equals physical CUDA device index",
        "files": render_files,
    }
    weights = []
    frozen_weights = []
    labels = ("native", "old_success_baseline_30k", "lora_only_30k", "joint_30k")
    for label in labels:
        adapter = tmp_path / f"{label}.pt"
        make_weight(results_root, label, adapter, render_backend)
        weights.extend(["--weight", label, label, str(adapter)])
        frozen_weights.append(
            {
                "label": label,
                "path": str(adapter.resolve()),
                "size_bytes": adapter.stat().st_size,
                "sha256": aggregate.sha256_file(adapter),
            }
        )
    snapshot_audit = results_root / "FROZEN_SNAPSHOT_PROTOCOL_AUDIT.json"
    write_json(
        snapshot_audit,
        {
            "status": "active",
            "evaluation_code_snapshot": {
                "path": str(evaluation_root),
                "git_commit": "test",
                "implementation_sha256": {
                    "shared.py": aggregate.sha256_file(evaluation_root / "shared.py")
                },
            },
            "official_protocol": {
                "initialization_steps": 5,
                "initialization_action": [0.0] * 7,
                "max_policy_steps": 600,
                "metric_script_sha256": aggregate.sha256_file(official_metric),
                "metric_script": str(official_metric),
                "eval_config": str(official_config),
                "eval_config_sha256": aggregate.sha256_file(official_config),
            },
            "render_backend": render_backend,
            "dependencies": {
                "base_model": str((results_root / "model").resolve()),
                "dataset_stats": {
                    "path": str((results_root / "stats.json").resolve()),
                    "sha256": aggregate.sha256_file(results_root / "stats.json"),
                },
                "policy_config": {
                    "path": str((results_root / "policy.json").resolve()),
                    "sha256": aggregate.sha256_file(results_root / "policy.json"),
                },
            },
            "ordered_weights": frozen_weights,
        },
    )
    libero_root = results_root / "libero"
    benchmark_root = libero_root / "libero" / "libero"
    task_catalog_path = benchmark_root / "benchmark" / "task_classification.json"
    write_json(task_catalog_path, {"synthetic": True})
    (benchmark_root / "bddl_files").mkdir()
    (benchmark_root / "init_files").mkdir()
    (benchmark_root / "bddl_files" / "task.bddl").write_text("bddl", encoding="utf-8")
    (benchmark_root / "init_files" / "task.pruned_init").write_text("init", encoding="utf-8")
    runtime_config = results_root / "libero-runtime.yaml"
    runtime_config.write_text("runtime", encoding="utf-8")
    resources = {}
    for name in ("benchmark", "bddl_files", "init_files"):
        resource_path = benchmark_root / name
        file_count, content_digest = aggregate.directory_content_digest(resource_path, benchmark_root)
        resources[name] = {
            "path": str(resource_path),
            "file_count": file_count,
            "content_digest": content_digest,
        }
    task_catalog_audit = results_root / "TASK_CATALOG_RESOURCE_AUDIT.json"
    write_json(
        task_catalog_audit,
        {
            "libero_plus": {
                "root": str(libero_root),
                "git_commit": "test-libero",
                "runtime_config": str(runtime_config),
                "runtime_config_sha256": aggregate.sha256_file(runtime_config),
            },
            "task_catalog": {
                "path": str(task_catalog_path),
                "sha256": aggregate.sha256_file(task_catalog_path),
                "total_tasks": 7,
                "suite_counts": {
                    "libero_spatial": 4,
                    "libero_object": 1,
                    "libero_goal": 1,
                    "libero_10": 1,
                },
            },
            "resource_snapshots": resources,
        },
    )
    monkeypatch.setattr(aggregate, "git_head", lambda path: "test-libero")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_fastwam_libero_shared_results.py",
            "--results-root",
            str(results_root),
            "--snapshot-audit",
            str(snapshot_audit),
            "--task-catalog-audit",
            str(task_catalog_audit),
            *weights,
        ],
    )
    aggregate.main()

    comparison = json.loads((results_root / "comparison_summary.json").read_text())
    assert comparison["verification"]["weights_verified"] == list(labels)
    assert comparison["verification"]["total_tasks_per_weight"] == 7
    assert comparison["verification"]["total_episodes_per_weight"] == 350
    assert comparison["verification"]["checkpoint_and_dependency_hashes_match_snapshot"]
    assert comparison["verification"]["evaluation_and_official_hashes_match_snapshot"]
    assert comparison["verification"]["libero_task_resources_match_baseline"]
    assert comparison["verification"]["episode_outcomes_match_official_horizon"]
    assert len(comparison["snapshot_audit"]["sha256"]) == 64
    assert len(comparison["task_catalog_audit"]["sha256"]) == 64
    assert comparison["weights"]["native"]["summary"]["overall"]["success_rate"] == 0.98
    assert comparison["weights"]["joint_30k"]["wall_clock"]["elapsed_seconds"] == 7.0
    native = comparison["weights"]["native"]
    assert native["checkpoint"]["path"] == str((tmp_path / "native.pt").resolve())
    assert len(native["checkpoint"]["sha256"]) == 64
    assert native["checkpoint"]["matches_snapshot_audit"]
    assert len(native["server_log_evidence"]) == 7
    assert all(item["commands_match_manifest"] for item in native["server_log_evidence"])
    assert all(item["action_infer_steps"] == 1 for item in native["server_log_evidence"])
    task = native["summary"]["tasks"]["libero_spatial/0"]
    assert task["failure_reasons"] == {"max_steps_exceeded": 1}
    assert task["average_policy_steps"] == 110.0
    assert task["rollout_elapsed_seconds"] == 50.0
    assert native["summary"]["categories"]["test"]["episodes"] == 350
    assert len((results_root / "comparison_summary.csv").read_text().splitlines()) == 53


def test_validates_twenty_step_shared_policy_weight(tmp_path):
    results_root = tmp_path / "results"
    adapter = tmp_path / "native.pt"
    make_weight(results_root, "native_20step", adapter, action_infer_steps=20)

    validated = aggregate.validate_weight(
        "native_20step",
        results_root / "native_20step",
        adapter,
        None,
        "official_protocol_shared_policy",
        expected_action_infer_steps=20,
        expected_actions_per_plan=10,
    )

    output = validated["output"]
    assert output["action_infer_steps"] == 20
    assert output["actions_per_plan"] == 10
    assert all(item["action_infer_steps"] == 20 for item in output["server_log_evidence"])


def test_five_model_spec_requires_requested_steps_and_actions(tmp_path):
    values = [
        [label, str(tmp_path / label), "protocol", str(tmp_path / f"{label}.pt"), str(steps), "10"]
        for label, steps in five_model.EXPECTED_MODELS.items()
    ]

    models = five_model.parse_models(values)
    assert {item["label"]: item["action_infer_steps"] for item in models} == (
        five_model.EXPECTED_MODELS
    )

    values[0][-2] = "1"
    with pytest.raises(ValueError, match="Unexpected action inference steps"):
        five_model.parse_models(values)


def test_verifies_five_models_with_different_inference_steps(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    (results_root / "stats.json").parent.mkdir(parents=True)
    (results_root / "stats.json").write_text("stats", encoding="utf-8")
    (results_root / "policy.json").write_text("policy", encoding="utf-8")
    native_adapter = tmp_path / "native.pt"
    adapters = {
        "native_20step": native_adapter,
        "native_1step": native_adapter,
        "joint_30k": tmp_path / "joint.pt",
        "lora_only_30k": tmp_path / "lora.pt",
        "old_joint_30k": tmp_path / "old-joint.pt",
    }
    values = []
    for label, steps in five_model.EXPECTED_MODELS.items():
        make_weight(
            results_root,
            label,
            adapters[label],
            action_infer_steps=steps,
        )
        values.append(
            [
                label,
                str(results_root / label),
                "official_protocol_shared_policy",
                str(adapters[label]),
                str(steps),
                "10",
            ]
        )

    task_counts = {
        "libero_spatial": 4,
        "libero_object": 1,
        "libero_goal": 1,
        "libero_10": 1,
    }
    monkeypatch.setattr(
        five_model.shared,
        "load_task_catalog_audit",
        lambda path: {
            "task_counts": task_counts,
            "total_tasks": 7,
            "output": {"path": str(path), "sha256": "synthetic"},
        },
    )

    weights, reference, task_catalog = five_model.validate_requested_models(
        five_model.parse_models(values), tmp_path / "catalog.json"
    )
    payload = five_model.build_payload(weights, reference, task_catalog)

    assert payload["verification"]["all_five_requested_models_complete"]
    assert payload["verification"]["same_episode_catalog_seed_and_initial_states"]
    assert payload["verification"]["total_episodes_per_model"] == 350
    assert payload["verification"]["total_episodes_all_models"] == 1750
    assert weights["native_20step"]["action_infer_steps"] == 20
    assert weights["native_1step"]["action_infer_steps"] == 1
    assert weights["native_20step"]["checkpoint"]["sha256"] == weights[
        "native_1step"
    ]["checkpoint"]["sha256"]
