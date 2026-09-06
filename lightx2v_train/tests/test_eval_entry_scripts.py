import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def run_dry_run(script_name, output_dir, **overrides):
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "OUTPUT_DIR": str(output_dir), **overrides})
    return subprocess.run(
        ["bash", str(ROOT / script_name)],
        cwd=output_dir.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


@pytest.mark.parametrize(
    ("script_name", "expected_fragments"),
    [
        (
            "robotwin_test.sh",
            (
                "/mnt/afs_1/charles/models/fastwam/robotwin_uncond_3cam_384.pt",
                "task=robotwin_uncond_3cam_384_1e-4",
                "EVALUATION.eval_num_episodes=100",
                "EVALUATION.num_inference_steps=10",
                "EVALUATION.replan_steps=24",
                "EVALUATION.instruction_type=unseen",
                "EVALUATION.skip_get_obs_within_replan=true",
                "MULTIRUN.num_gpus=8",
                "+MULTIRUN.gpu_ids=\\[0\\,1\\,2\\,3\\,4\\,5\\,6\\,7\\]",
                "MULTIRUN.max_tasks_per_gpu=1",
            ),
        ),
        (
            "libero_test.sh",
            (
                "eval_fastwam_libero_checkpoint.py",
                "checkpoint-000030000-student.pt",
                "--benchmarks libero_spatial libero_object libero_goal libero_10",
                "--devices 0 1 2 3 4 5 6 7",
                "--episodes-per-task 50",
                "--tasks-per-shard 5",
                "--expected-action-infer-steps 1",
            ),
        ),
        (
            "libero_plus_test.sh",
            (
                "LightX2V_fastwam_20step_f8164573/lightx2v_train/tools/eval_fastwam_libero_shared_checkpoint.py",
                "/mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt",
                "--devices 0 1 2 3 4 5 6 7",
                "--env-workers-per-device 12",
                "--egl-device-override 5=4",
                "--egl-device-override 7=6",
                "--episodes-per-task 1",
                "--tasks-per-shard 1",
                "--expected-action-infer-steps 20",
                "--expected-actions-per-plan 10",
                "--nvidia-egl-root /mnt/afs_1/charles/env/nvidia-egl-550.90.07/root",
            ),
        ),
    ],
)
def test_default_dry_run_builds_full_evaluation_command(tmp_path, script_name, expected_fragments):
    output_dir = tmp_path / script_name.removesuffix(".sh")

    result = run_dry_run(script_name, output_dir)

    assert result.returncode == 0, result.stdout
    assert output_dir.is_dir()
    assert "Mode: DRY_RUN" in result.stdout
    assert f"Output directory: {output_dir}" in result.stdout
    assert "Environment:" in result.stdout
    assert "Command:" in result.stdout
    for fragment in expected_fragments:
        assert fragment in result.stdout
    assert "ROBOTWIN_FORCE_RASTER" not in result.stdout
    assert "lavapipe" not in result.stdout.lower()
    assert not (output_dir / "manager.log").exists()
    assert not (output_dir / "driver.log").exists()


def test_environment_overrides_reach_libero_command(tmp_path):
    output_dir = tmp_path / "custom-libero"

    result = run_dry_run(
        "libero_test.sh",
        output_dir,
        EPISODES_PER_TASK="3",
        GPU_IDS="2 4",
        TASKS_PER_SHARD="2",
    )

    assert result.returncode == 0, result.stdout
    assert "--devices 2 4" in result.stdout
    assert "--episodes-per-task 3" in result.stdout
    assert "--tasks-per-shard 2" in result.stdout


def test_missing_weight_has_clear_error(tmp_path):
    result = run_dry_run(
        "robotwin_test.sh",
        tmp_path / "missing-weight",
        WEIGHT=str(tmp_path / "does-not-exist.pt"),
    )

    assert result.returncode != 0
    assert "ERROR: weight file not found:" in result.stdout
    assert str(tmp_path / "does-not-exist.pt") in result.stdout
