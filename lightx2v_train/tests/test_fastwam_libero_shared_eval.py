import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import eval_fastwam_libero_shared_checkpoint as shared_checkpoint
import eval_fastwam_libero_shared_policy as shared_eval


class FakeObserver:
    task_description = "move the object"

    def __init__(self, policy_steps_until_success):
        self.policy_steps_until_success = policy_steps_until_success
        self.total_steps = 0

    def reset(self):
        self.total_steps = 0
        return self._observation()

    def step(self, action):
        assert np.asarray(action).shape == (7,)
        self.total_steps += 1
        success = self.total_steps >= 5 + self.policy_steps_until_success
        return self._observation(), 0.0, success, {}

    @staticmethod
    def _observation():
        return {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }


class ImmediateRequestQueue:
    def __init__(self, response_queue):
        self.response_queue = response_queue
        self.requests = []

    def put(self, request):
        self.requests.append(request)
        _, request_id, _, _, _ = request
        actions = np.zeros((32, 7), dtype=np.float32)
        self.response_queue.put((request_id, actions, None))


def test_shared_episode_matches_official_initialization_and_chunk_consumption():
    response_queue = queue.Queue()
    request_queue = ImmediateRequestQueue(response_queue)
    observer = FakeObserver(policy_steps_until_success=12)

    success, steps, _ = shared_eval.run_shared_episode(
        observer,
        request_queue,
        response_queue,
        worker_index=3,
        request_counter=shared_eval.counter(),
        actions_per_plan=10,
        initialization_steps=5,
        max_steps=600,
    )

    assert success is True
    assert steps == 12
    assert observer.total_steps == 17
    assert len(request_queue.requests) == 2
    assert all(request[0] == 3 for request in request_queue.requests)


def test_complete_shard_requires_exact_signature_catalog_and_protocol(tmp_path):
    args = SimpleNamespace(
        adapter=str(tmp_path / "adapter.pt"),
        config=str(tmp_path / "config.json"),
        model_path=str(tmp_path / "model"),
        dataset_stats=str(tmp_path / "stats.json"),
        libero_root=str(tmp_path / "libero"),
        episodes_per_task=2,
        episode_offset=0,
        max_steps=0,
        render_size=256,
        seed=0,
        expected_action_infer_steps=1,
        prompt_cache_limit=32,
    )
    shard = {"name": "libero_spatial-tasks-00000-00000", "benchmark": "libero_spatial", "task_ids": (0,)}
    official = {"initialization_steps": 5, "max_policy_steps": 600}
    implementation = {"shared.py": {"path": "/shared.py", "sha256": "abc"}}
    signature = shared_eval.shard_run_signature(args, shard, official, implementation)
    signature["shared_policy"]["actions_per_plan"] = 10
    episodes = [
        {"benchmark": "libero_spatial", "task_id": 0, "episode_index": episode_index}
        for episode_index in range(2)
    ]
    payload = {
        "finished_at": "2026-08-29T00:00:00Z",
        "run_signature": signature,
        "protocol": {
            "action_infer_steps": 1,
            "actions_per_plan": 10,
            "seed": 0,
            "official_evaluation": official,
            "fastwam_evaluation_implementation": implementation,
        },
        "episodes": episodes,
    }
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert shared_eval.shard_is_complete(path, shard, args, official, implementation, 10)

    payload["protocol"]["seed"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not shared_eval.shard_is_complete(path, shard, args, official, implementation, 10)

    payload["protocol"]["seed"] = 0
    payload["run_signature"]["render_environment"] = {"mujoco_gl": "egl"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert shared_eval.shard_is_complete(
        path,
        shard,
        args,
        official,
        implementation,
        10,
        {"mujoco_gl": "egl"},
    )


def test_shard_signature_tracks_render_environment(tmp_path, monkeypatch):
    args = SimpleNamespace(
        adapter=str(tmp_path / "adapter.pt"),
        config=str(tmp_path / "config.json"),
        model_path=str(tmp_path / "model"),
        dataset_stats=str(tmp_path / "stats.json"),
        libero_root=str(tmp_path / "libero"),
        episodes_per_task=1,
        episode_offset=0,
        max_steps=0,
        render_size=256,
        seed=0,
        expected_action_infer_steps=1,
        prompt_cache_limit=32,
    )
    shard = {"benchmark": "libero_spatial", "task_ids": (0,)}
    monkeypatch.setenv("NVIDIA_EGL_ROOT", "/egl-a")
    first = shared_eval.shard_run_signature(args, shard, {}, {})
    monkeypatch.setenv("NVIDIA_EGL_ROOT", "/egl-b")
    second = shared_eval.shard_run_signature(args, shard, {}, {})
    assert first != second
    assert first["render_environment"]["nvidia_egl_root"] == "/egl-a"


def test_nvidia_egl_server_environment_uses_physical_device(tmp_path, monkeypatch):
    root = tmp_path / "egl"
    library_dir = root / "usr" / "lib" / "x86_64-linux-gnu"
    vendor_dir = root / "usr" / "share" / "glvnd" / "egl_vendor.d"
    library_dir.mkdir(parents=True)
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "10_nvidia.json").write_text(
        json.dumps({"ICD": {"library_path": "libEGL_nvidia.so.0"}}), encoding="utf-8"
    )
    for name in ("libEGL_nvidia.so.0", "libnvidia-eglcore.so.1", "libnvidia-glsi.so.1"):
        (library_dir / name).write_text(name, encoding="utf-8")
    runtime = shared_checkpoint.resolve_nvidia_egl_runtime(root)
    args = SimpleNamespace(nvidia_egl_runtime=runtime)
    item = {
        "cuda_visible_devices": "7",
        "mujoco_egl_device_id": "7",
    }
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    env = shared_checkpoint.server_environment(args, item)
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "7"
    assert env["EGL_PLATFORM"] == "surfaceless"
    assert env["LD_LIBRARY_PATH"].split(":") == [str(library_dir), "/existing"]
    assert env["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(vendor_dir / "10_nvidia.json")

    command_args = SimpleNamespace(
        policy_config=tmp_path / "policy.json",
        adapter=tmp_path / "adapter.pt",
        model_path=tmp_path / "model",
        dataset_stats=tmp_path / "stats.json",
        libero_root=tmp_path / "libero",
        benchmarks=["libero_spatial"],
        episodes_per_task=1,
        episode_offset=0,
        tasks_per_shard=1,
        render_size=256,
        seed=0,
        expected_action_infer_steps=1,
        expected_actions_per_plan=10,
        env_workers_per_device=1,
        env_workers_by_device={7: 1},
        prompt_cache_limit=1,
        task_ids=[0],
        max_steps=0,
        devices=[7],
        nvidia_egl_runtime=runtime,
    )
    command, visible_devices, egl_device_id = shared_checkpoint.server_command(
        command_args, 7, 0, tmp_path / "output", tmp_path / "ready.json"
    )
    assert visible_devices == "7"
    assert egl_device_id == "7"
    assert command[command.index("--env-workers") + 1] == "1"


def test_per_device_environment_worker_overrides():
    workers = shared_checkpoint.resolve_env_workers_by_device(
        devices=[1, 2, 5],
        default_count=12,
        overrides=["5=1"],
    )
    assert workers == {1: 12, 2: 12, 5: 1}
