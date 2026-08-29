import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

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
