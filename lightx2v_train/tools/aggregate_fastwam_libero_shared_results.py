"""Strictly verify and compare four shared-policy LIBERO-plus evaluations."""

import argparse
import hashlib
import json
import locale
import subprocess
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
    parser.add_argument(
        "--snapshot-audit",
        help="Frozen pre-run audit containing expected checkpoint and dependency SHA256 values",
    )
    parser.add_argument(
        "--task-catalog-audit",
        help="Baseline audit for the LIBERO-plus task catalog, BDDL, and init-state resources",
    )
    return parser.parse_args()


def parse_timestamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_file(name, expected):
    path = Path(expected.get("path", "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {name}: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected.get("sha256"):
        raise RuntimeError(f"{name} SHA256 differs from frozen audit: {path}")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "matches_snapshot_audit": True,
    }


def directory_content_digest(path, relative_root):
    path = Path(path).expanduser().resolve()
    relative_root = Path(relative_root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing frozen resource directory: {path}")
    locale.setlocale(locale.LC_COLLATE, "zh_CN.UTF-8")
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: locale.strxfrm(item.relative_to(relative_root).as_posix()),
    )
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(relative_root).as_posix()
        if "\\" in relative or "\n" in relative:
            raise RuntimeError(f"Unsupported resource path for sha256sum-compatible digest: {item}")
        digest.update(f"{sha256_file(item)}  {relative}\n".encode("utf-8"))
    return len(files), digest.hexdigest()


def git_head(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_task_catalog_audit(path):
    if not path:
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing task catalog resource audit: {path}")
    payload = common.load_json(path)
    libero = payload.get("libero_plus", {})
    libero_root = Path(libero.get("root", "")).expanduser().resolve()
    expected_commit = libero.get("git_commit")
    actual_commit = git_head(libero_root)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"LIBERO-plus commit differs from task catalog audit: {actual_commit} != {expected_commit}"
        )

    runtime_config = verified_file(
        "LIBERO runtime config",
        {
            "path": libero.get("runtime_config"),
            "sha256": libero.get("runtime_config_sha256"),
        },
    )
    catalog = verified_file("LIBERO task catalog", payload.get("task_catalog", {}))
    benchmark_root = libero_root / "libero" / "libero"
    resources = {}
    for name in ("benchmark", "bddl_files", "init_files"):
        expected = payload.get("resource_snapshots", {}).get(name, {})
        resource_path = Path(expected.get("path", "")).expanduser().resolve()
        file_count, content_digest = directory_content_digest(resource_path, benchmark_root)
        if file_count != int(expected.get("file_count", -1)):
            raise RuntimeError(
                f"{name} file count differs from task catalog audit: "
                f"{file_count} != {expected.get('file_count')}"
            )
        if content_digest != expected.get("content_digest"):
            raise RuntimeError(f"{name} content digest differs from task catalog audit: {resource_path}")
        resources[name] = {
            "path": str(resource_path),
            "file_count": file_count,
            "content_digest": content_digest,
            "matches_task_catalog_audit": True,
        }
    return {
        "task_counts": payload["task_catalog"]["suite_counts"],
        "total_tasks": int(payload["task_catalog"]["total_tasks"]),
        "output": {
            "path": str(path),
            "sha256": sha256_file(path),
            "libero_plus_root": str(libero_root),
            "libero_plus_git_commit": actual_commit,
            "runtime_config": runtime_config,
            "task_catalog": catalog,
            "resources": resources,
        },
    }


def checkpoint_evidence(adapter, expected=None):
    path = Path(adapter)
    if not path.is_file():
        raise FileNotFoundError(f"Missing adapter checkpoint: {path}")
    evidence = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if expected is not None:
        if common.resolved(expected["path"]) != evidence["path"]:
            raise RuntimeError(f"Checkpoint path differs from frozen audit: {path}")
        if int(expected["size_bytes"]) != evidence["size_bytes"]:
            raise RuntimeError(f"Checkpoint size differs from frozen audit: {path}")
        if expected["sha256"] != evidence["sha256"]:
            raise RuntimeError(f"Checkpoint SHA256 differs from frozen audit: {path}")
        evidence["expected_sha256"] = expected["sha256"]
        evidence["matches_snapshot_audit"] = True
    return evidence


def load_snapshot_audit(path):
    if not path:
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen snapshot audit: {path}")
    payload = common.load_json(path)
    weights = {item["label"]: item for item in payload.get("ordered_weights", [])}
    dependencies = payload.get("dependencies", {})
    dataset_stats = dependencies.get("dataset_stats", {})
    policy_config = dependencies.get("policy_config", {})
    required_labels = {"native", "old_success_baseline_30k", "lora_only_30k", "joint_30k"}
    if set(weights) != required_labels:
        raise RuntimeError(f"Frozen snapshot audit has unexpected weights: {sorted(weights)}")

    verified_dependencies = {}
    for name, expected in (("dataset_stats", dataset_stats), ("policy_config", policy_config)):
        verified_dependencies[name] = verified_file(name, expected)

    official = payload.get("official_protocol", {})
    verified_official = {
        "metric_script": verified_file(
            "official metric script",
            {"path": official.get("metric_script"), "sha256": official.get("metric_script_sha256")},
        ),
        "eval_config": verified_file(
            "official eval config",
            {"path": official.get("eval_config"), "sha256": official.get("eval_config_sha256")},
        ),
    }
    evaluation_snapshot = payload.get("evaluation_code_snapshot", {})
    evaluation_root = Path(evaluation_snapshot.get("path", "")).expanduser().resolve()
    verified_implementation = {}
    for relative_path, expected_sha256 in evaluation_snapshot.get("implementation_sha256", {}).items():
        verified_implementation[relative_path] = verified_file(
            f"evaluation implementation {relative_path}",
            {"path": evaluation_root / relative_path, "sha256": expected_sha256},
        )
    if not verified_implementation:
        raise RuntimeError("Frozen snapshot audit has no evaluation implementation hashes")

    render_backend = payload.get("render_backend")
    verified_render_files = {}
    if render_backend is not None:
        if render_backend.get("name") != "nvidia-egl":
            raise RuntimeError(f"Unexpected frozen render backend: {render_backend.get('name')}")
        for name, expected in render_backend.get("files", {}).items():
            verified_render_files[name] = verified_file(f"render backend {name}", expected)
        required_render_files = {
            "egl_vendor_json",
            "libegl_nvidia",
            "libnvidia_eglcore",
            "libnvidia_glsi",
        }
        if set(verified_render_files) != required_render_files:
            raise RuntimeError(
                f"Frozen render backend has unexpected files: {sorted(verified_render_files)}"
            )

    base_model = common.resolved(dependencies["base_model"])
    if not Path(base_model).exists():
        raise FileNotFoundError(f"Missing frozen base model: {base_model}")
    return {
        "weights": weights,
        "base_model": base_model,
        "dependencies": verified_dependencies,
        "official": official,
        "render_backend": render_backend,
        "output": {
            "path": str(path),
            "sha256": sha256_file(path),
            "status": payload.get("status"),
            "evaluation_code_snapshot": {
                "path": str(evaluation_root),
                "git_commit": evaluation_snapshot.get("git_commit"),
                "implementation": verified_implementation,
            },
            "official_protocol_files": verified_official,
            "render_backend": (
                {**render_backend, "files": verified_render_files}
                if render_backend is not None
                else None
            ),
            "dependencies": {"base_model": base_model, **verified_dependencies},
            "weight_sha256": {label: item["sha256"] for label, item in weights.items()},
        },
    }


def validate_manifest_against_snapshot(manifest, snapshot):
    if snapshot is None:
        return
    expected = {
        "model_path": snapshot["base_model"],
        "dataset_stats": snapshot["dependencies"]["dataset_stats"]["path"],
        "policy_config": snapshot["dependencies"]["policy_config"]["path"],
    }
    mismatches = {
        key: {"actual": common.resolved(manifest[key]), "expected": value}
        for key, value in expected.items()
        if common.resolved(manifest[key]) != value
    }
    if mismatches:
        raise RuntimeError(f"Command manifest differs from frozen dependency paths: {mismatches}")
    manifest_official = manifest.get("official_evaluation", {})
    official_mismatches = {
        key: {"actual": value, "expected": snapshot["official"].get(key)}
        for key, value in manifest_official.items()
        if snapshot["official"].get(key) != value
    }
    if official_mismatches:
        raise RuntimeError(
            f"Command manifest differs from frozen official protocol: {official_mismatches}"
        )
    if snapshot["render_backend"] is not None and manifest.get("render_backend") != snapshot[
        "render_backend"
    ]:
        raise RuntimeError("Command manifest differs from frozen render backend")


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


def validate_episode_outcome(label, key, episode):
    initialization_steps = int(episode["initialization_steps"])
    policy_steps = int(episode["policy_steps"])
    max_policy_steps = int(episode["max_policy_steps"])
    if not 1 <= policy_steps <= max_policy_steps:
        raise RuntimeError(f"{label} episode {key} has policy steps outside the official horizon")
    if int(episode["total_env_steps"]) != initialization_steps + policy_steps:
        raise RuntimeError(f"{label} episode {key} has inconsistent total environment steps")
    if bool(episode["success"]):
        if episode.get("failure_reason") is not None:
            raise RuntimeError(f"{label} successful episode {key} records a failure reason")
    elif policy_steps != max_policy_steps or episode.get("failure_reason") != "max_steps_exceeded":
        raise RuntimeError(f"{label} failed episode {key} did not exhaust the official horizon")


def expected_render_environment(manifest, command):
    backend = manifest.get("render_backend", {})
    if backend.get("name") != "nvidia-egl":
        return None
    return {
        "mujoco_gl": "egl",
        "pyopengl_platform": "egl",
        "egl_platform": "surfaceless",
        "cuda_visible_devices": command["cuda_visible_devices"],
        "mujoco_egl_device_id": command["mujoco_egl_device_id"],
        "nvidia_egl_root": backend["root"],
        "__egl_vendor_library_filenames": backend["files"]["egl_vendor_json"]["path"],
    }


def validate_shards(protocol_root, expected_shards, official, implementation, manifest):
    paths = sorted((protocol_root / "shards").glob("*.json"))
    if len(paths) != expected_shards:
        raise RuntimeError(f"{protocol_root} has {len(paths)} shards, expected {expected_shards}")
    starts = []
    finishes = []
    physical_devices = set()
    commands = {int(item["physical_cuda_device"]): item for item in manifest["commands"]}
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
        physical_device = int(protocol["physical_cuda_device"])
        expected_render = expected_render_environment(manifest, commands[physical_device])
        if expected_render is not None and protocol.get("render_environment") != expected_render:
            raise RuntimeError(f"NVIDIA EGL render environment mismatch in {path}")
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


def validate_weight(label, result_dir, expected_adapter, reference, protocol_directory, snapshot=None):
    protocol_root = result_dir / protocol_directory
    summary_path = protocol_root / "summary.json"
    commands_path = protocol_root / "commands.json"
    if not summary_path.is_file() or not commands_path.is_file():
        raise FileNotFoundError(f"Missing completed outputs for {label}: {summary_path}, {commands_path}")
    result = common.load_json(summary_path)
    manifest = common.load_json(commands_path)
    adapter = common.resolved(expected_adapter)
    validate_manifest_against_snapshot(manifest, snapshot)
    expected_checkpoint = snapshot["weights"][label] if snapshot is not None else None

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
        validate_episode_outcome(label, key, episode)
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

    wall_clock = validate_shards(
        protocol_root, expected_shards, official, implementation, manifest
    )
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
            "checkpoint": checkpoint_evidence(adapter, expected_checkpoint),
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
    snapshot = load_snapshot_audit(args.snapshot_audit)
    task_catalog = load_task_catalog_audit(args.task_catalog_audit)
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
            snapshot,
        )
        if reference is None:
            reference = validated["reference"]
        weights[label] = validated["output"]

    if task_catalog is not None:
        if reference["task_counts"] != task_catalog["task_counts"]:
            raise RuntimeError("Completed suite task counts differ from the task catalog audit")
        if sum(reference["task_counts"].values()) != task_catalog["total_tasks"]:
            raise RuntimeError("Completed task total differs from the task catalog audit")

    payload = {
        "generated_at": common.utc_now(),
        "results_root": str(results_root),
        "snapshot_audit": snapshot["output"] if snapshot is not None else None,
        "task_catalog_audit": task_catalog["output"] if task_catalog is not None else None,
        "verification": {
            "weights_verified": list(weights),
            "all_suites_and_tasks_complete": True,
            "same_episode_catalog_seed_and_initial_states": True,
            "official_protocol_and_implementation_match": True,
            "episode_outcomes_match_official_horizon": True,
            "shared_policy_lazy_prompt_cache_verified": True,
            "checkpoint_and_dependency_hashes_match_snapshot": snapshot is not None,
            "evaluation_and_official_hashes_match_snapshot": snapshot is not None,
            "libero_task_resources_match_baseline": task_catalog is not None,
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
