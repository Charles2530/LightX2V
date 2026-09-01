#!/usr/bin/env python3
"""Watch LightX2V FastWAM RobotWin DMD checkpoints and evaluate them.

This utility intentionally does not install or modify Python environments.  It
uses the LightX2V venv only for exporting action-DMD LoRA checkpoints and the
RoboTwin conda environment only for simulator evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIGHTX2V_ROOT = Path("/mnt/afs_1/charles/codes/LightX2V_fastwam")
LIGHTX2V_TRAIN_ROOT = LIGHTX2V_ROOT / "lightx2v_train"
FASTWAM_ROOT = Path("/mnt/afs_1/charles/codes/FastWAM")
ROBOTWIN_PYTHON = Path("/mnt/afs_1/charles/env/miniconda3/envs/robotwin/bin/python")
LIGHTX2V_PYTHON = LIGHTX2V_ROOT / ".venv/bin/python"

DEFAULT_RUN_DIR = LIGHTX2V_TRAIN_ROOT / "runs/fastwam_robotwin_action_1step_dmd_lora_only"
DEFAULT_TRAIN_CONFIG = (
    LIGHTX2V_TRAIN_ROOT
    / "configs/train/fastwam_action_dmd/robotwin_action_1step_dmd_lora_only.yaml"
)
DEFAULT_ORIGINAL_CKPT = Path("/mnt/afs_1/charles/models/fastwam/robotwin_uncond_3cam_384.pt")
DEFAULT_ORIGINAL_STATS = Path(
    "/mnt/afs_1/charles/models/fastwam/robotwin_uncond_3cam_384_dataset_stats.json"
)
DEFAULT_EVAL_ROOT = DEFAULT_RUN_DIR / "robotwin_eval"
DEFAULT_EXPORT_DIR = (
    FASTWAM_ROOT
    / "evaluate_results/robotwin_lightx2v_exports/fastwam_robotwin_action_1step_dmd_lora_only"
)
DEFAULT_WAN22_MODEL_ROOT = Path("/mnt/afs_1/charles/models/Wan2.2-TI2V-5B")
DEFAULT_WAN21_TOKENIZER_ROOT = Path("/mnt/afs_1/charles/models/Wan2.1-T2V-1.3B")
DEFAULT_SAPIEN_ICD = (
    ROBOTWIN_PYTHON.parents[1]
    / "lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json"
)
DEFAULT_LOCAL_NVIDIA_ROOT = DEFAULT_EVAL_ROOT / "nvidia_gl_550_extracted"
DEFAULT_LOCAL_NVIDIA_LIB_DIR = DEFAULT_LOCAL_NVIDIA_ROOT / "usr/lib/x86_64-linux-gnu"
DEFAULT_LOCAL_NVIDIA_ICD = DEFAULT_LOCAL_NVIDIA_ROOT / "nvidia_icd_abs.json"
DEFAULT_LOCAL_EGL_VENDOR = DEFAULT_LOCAL_NVIDIA_ROOT / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
DEFAULT_LOCAL_VULKAN_LOADER_LIB_DIR = (
    DEFAULT_EVAL_ROOT / "mesa_vulkan_extracted/usr/lib/x86_64-linux-gnu"
)

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
ITER_RE = re.compile(r"\[train\] iter=(\d+)/(\d+)")
FINISHED_RE = re.compile(r"\[train\] finished .* iter=(\d+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class Recorder:
    def __init__(self, records_path: Path, log_path: Path) -> None:
        self.records_path = records_path
        self.log_path = log_path
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, **payload: Any) -> None:
        record = {"ts": utc_now(), "event": event, **payload}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{record['ts']} {event} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n")
        print(line, flush=True)


def first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NVIDIA_VISIBLE_DEVICES"] = args.nvidia_visible_devices
    env.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

    auto_nvidia_root = args.eval_root / "nvidia_gl_550_extracted"
    auto_nvidia_lib_dir = auto_nvidia_root / "usr/lib/x86_64-linux-gnu"
    auto_nvidia_icd = auto_nvidia_root / "nvidia_icd_abs.json"
    auto_egl_vendor = auto_nvidia_root / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    auto_vulkan_loader_lib_dir = args.eval_root / "mesa_vulkan_extracted/usr/lib/x86_64-linux-gnu"

    vk_icd = (
        expand(args.vk_icd_filenames)
        if args.vk_icd_filenames
        else first_existing(auto_nvidia_icd, DEFAULT_LOCAL_NVIDIA_ICD, DEFAULT_SAPIEN_ICD)
    )
    if vk_icd:
        env["VK_ICD_FILENAMES"] = str(vk_icd)

    egl_vendor = (
        expand(args.egl_vendor_library_filenames)
        if args.egl_vendor_library_filenames
        else first_existing(auto_egl_vendor, DEFAULT_LOCAL_EGL_VENDOR)
    )
    if egl_vendor:
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl_vendor)
    if args.glx_vendor_library_name:
        env["__GLX_VENDOR_LIBRARY_NAME"] = args.glx_vendor_library_name

    conda_lib = str(args.robotwin_python.parents[1] / "lib")
    extra_ld_paths: list[str] = []
    vulkan_loader_lib_dir = (
        expand(args.vulkan_loader_lib_dir)
        if args.vulkan_loader_lib_dir
        else first_existing(auto_vulkan_loader_lib_dir, DEFAULT_LOCAL_VULKAN_LOADER_LIB_DIR)
    )
    if vulkan_loader_lib_dir:
        extra_ld_paths.append(str(vulkan_loader_lib_dir))
    nvidia_lib_dir = (
        expand(args.nvidia_driver_lib_dir)
        if args.nvidia_driver_lib_dir
        else first_existing(auto_nvidia_lib_dir, DEFAULT_LOCAL_NVIDIA_LIB_DIR)
    )
    if nvidia_lib_dir:
        extra_ld_paths.append(str(nvidia_lib_dir))
    extra_ld_paths.append(conda_lib)
    old_ld = env.get("LD_LIBRARY_PATH", "")
    if old_ld:
        extra_ld_paths.append(old_ld)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(extra_ld_paths)
    py_paths = [str(args.fastwam_root / "src"), str(args.fastwam_root), str(args.lightx2v_root)]
    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        py_paths.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    return env


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    recorder: Recorder,
    event_prefix: str,
    timeout: int | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recorder.event(f"{event_prefix}_start", cmd=cmd, cwd=str(cwd), log=str(log_path))
    with log_path.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            recorder.event(f"{event_prefix}_timeout", timeout=timeout, log=str(log_path))
            return 124
    recorder.event(f"{event_prefix}_exit", return_code=return_code, log=str(log_path))
    return return_code


def tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_chars:].decode("utf-8", errors="replace")


def probe_render(args: argparse.Namespace, recorder: Recorder) -> bool:
    code = r"""
import traceback
try:
    import sapien.core as sapien
    from sapien.render import set_global_config
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    print("sapien render setup OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(2)
"""
    env = command_env(args)
    cmd = [str(args.robotwin_python), "-c", code]
    proc = subprocess.run(
        cmd,
        cwd=str(args.fastwam_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
        check=False,
    )
    if proc.returncode == 0:
        recorder.event("render_probe_ok")
        return True
    recorder.event(
        "render_probe_failed",
        return_code=proc.returncode,
        output=proc.stdout[-3000:],
        nvidia_visible_devices=env.get("NVIDIA_VISIBLE_DEVICES"),
        vk_icd_filenames=env.get("VK_ICD_FILENAMES"),
        egl_vendor_library_filenames=env.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
        glx_vendor_library_name=env.get("__GLX_VENDOR_LIBRARY_NAME"),
    )
    return False


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def list_target_checkpoints(args: argparse.Namespace) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    if not args.train_run_dir.exists():
        return items
    for path in args.train_run_dir.iterdir():
        if not path.is_dir():
            continue
        step = checkpoint_step(path)
        if step is None:
            continue
        if step <= 0 or step % args.step_interval != 0:
            continue
        if args.max_step is not None and step > args.max_step:
            continue
        if not checkpoint_ready(path, args.ready_age_seconds):
            continue
        items.append((step, path))
    return sorted(items)


def checkpoint_ready(path: Path, ready_age_seconds: int) -> bool:
    required = [path / "student_action.pt", path / "config.yaml"]
    if not all(item.exists() and item.stat().st_size > 0 for item in required):
        return False
    newest_mtime = max(item.stat().st_mtime for item in required)
    return time.time() - newest_mtime >= ready_age_seconds


def parse_train_progress(train_log: Path) -> dict[str, int | bool | None]:
    progress: dict[str, int | bool | None] = {
        "last_iter": None,
        "max_iter": None,
        "finished": False,
        "finished_iter": None,
    }
    text = tail_text(train_log, max_chars=200000)
    for match in ITER_RE.finditer(text):
        progress["last_iter"] = int(match.group(1))
        progress["max_iter"] = int(match.group(2))
    for match in FINISHED_RE.finditer(text):
        progress["finished"] = True
        progress["finished_iter"] = int(match.group(1))
    return progress


def export_checkpoint(step: int, checkpoint_dir: Path, args: argparse.Namespace, recorder: Recorder) -> Path | None:
    output = args.export_dir / f"lightx2v_robotwin_action_dmd_step_{step:09d}.pt"
    if output.exists() and output.stat().st_size > 0:
        recorder.event("export_skip_existing", step=step, output=str(output))
        return output

    cmd = [
        str(args.lightx2v_python),
        str(args.lightx2v_train_root / "tools/export_fastwam_action_dmd.py"),
        "--config",
        str(args.train_config),
        "--checkpoint",
        str(checkpoint_dir),
        "--output",
        str(output),
    ]
    env = command_env(args)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.lightx2v_root), str(args.lightx2v_train_root), env.get("PYTHONPATH", "")]
    )
    log_path = args.eval_root / "logs" / f"export_step_{step:09d}.log"
    code = run_logged(
        cmd,
        cwd=args.lightx2v_train_root,
        env=env,
        log_path=log_path,
        recorder=recorder,
        event_prefix="export",
    )
    if code != 0:
        recorder.event("export_failed", step=step, checkpoint=str(checkpoint_dir), log=str(log_path))
        return None
    if not output.exists() or output.stat().st_size == 0:
        recorder.event("export_missing_output", step=step, output=str(output), log=str(log_path))
        return None
    recorder.event("export_done", step=step, output=str(output), bytes=output.stat().st_size)
    return output


def manager_result_dir(args: argparse.Namespace, ckpt: Path, run_tag: str) -> Path:
    return args.fastwam_root / "evaluate_results" / "robotwin" / ckpt.stem / run_tag


def parse_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    payload = read_json(summary_path, {})
    overall = payload.get("overall") if isinstance(payload, dict) else None
    if not isinstance(overall, dict):
        return payload
    return {
        "summary_path": str(summary_path),
        "clean_mean_success_rate": overall.get("clean_mean_success_rate"),
        "random_mean_success_rate": overall.get("random_mean_success_rate"),
    }


def run_robotwin_manager(
    *,
    ckpt: Path,
    dataset_stats: Path,
    episodes: int,
    run_tag: str,
    args: argparse.Namespace,
    recorder: Recorder,
    kind: str,
    step: int | None = None,
) -> bool:
    cmd = [
        str(args.robotwin_python),
        "-u",
        "experiments/robotwin/run_robotwin_manager.py",
        "task=robotwin_uncond_3cam_384_1e-4",
        f"ckpt={ckpt}",
        f"EVALUATION.dataset_stats_path={dataset_stats}",
        f"EVALUATION.eval_num_episodes={episodes}",
        f"EVALUATION.num_inference_steps={args.num_inference_steps}",
        f"EVALUATION.replan_steps={args.replan_steps}",
        f"EVALUATION.output_dir=./evaluate_results/robotwin_monitor/{run_tag}",
        f"MULTIRUN.num_gpus={args.num_gpus}",
        f"MULTIRUN.max_tasks_per_gpu={args.max_tasks_per_gpu}",
        f"model.model_id={args.wan22_model_root}",
        f"model.tokenizer_model_id={args.wan21_tokenizer_root}",
        "model.redirect_common_files=false",
        "model.skip_dit_load_from_pretrain=true",
    ]
    if args.task_name:
        cmd.append(f"EVALUATION.task_name={args.task_name}")
    cmd.extend(args.manager_overrides)

    env = command_env(args)
    log_path = args.eval_root / "logs" / f"{kind}_{run_tag}.log"
    code = run_logged(
        cmd,
        cwd=args.fastwam_root,
        env=env,
        log_path=log_path,
        recorder=recorder,
        event_prefix=f"{kind}_eval",
    )
    result_dir = manager_result_dir(args, ckpt, run_tag)
    summary_path = result_dir / "summary.json"
    summary = parse_summary(summary_path)
    if code == 0 and summary is not None:
        recorder.event(
            f"{kind}_eval_done",
            step=step,
            ckpt=str(ckpt),
            episodes=episodes,
            num_inference_steps=args.num_inference_steps,
            result_dir=str(result_dir),
            summary=summary,
        )
        return True

    recorder.event(
        f"{kind}_eval_failed",
        step=step,
        ckpt=str(ckpt),
        episodes=episodes,
        return_code=code,
        result_dir=str(result_dir),
        summary_path=str(summary_path),
        log=str(log_path),
        log_tail=tail_text(log_path, 3000),
    )
    return False


def run_baseline(args: argparse.Namespace, state: dict[str, Any], recorder: Recorder) -> None:
    baseline = state.setdefault("baseline", {})
    if (
        baseline.get("status") == "success"
        and int(baseline.get("episodes", -1)) == args.baseline_num_episodes
        and int(baseline.get("num_inference_steps", -1)) == args.num_inference_steps
        and not args.force
    ):
        recorder.event("baseline_skip_success", summary=baseline.get("summary"))
        return

    if not should_retry(baseline, args.render_retry_seconds) and not args.force:
        return

    if args.probe_render and not probe_render(args, recorder):
        baseline.update(
            {
                "status": "pending_render_unavailable",
                "last_attempt_ts": utc_now(),
                "episodes": args.baseline_num_episodes,
            }
        )
        write_json(args.state_path, state)
        recorder.event("baseline_wait_render", retry_seconds=args.render_retry_seconds)
        return

    run_tag = (
        f"original_{args.baseline_num_episodes}trials_"
        f"{args.num_inference_steps}step_{compact_timestamp()}"
    )
    ok = run_robotwin_manager(
        ckpt=args.original_ckpt,
        dataset_stats=args.original_dataset_stats,
        episodes=args.baseline_num_episodes,
        run_tag=run_tag,
        args=args,
        recorder=recorder,
        kind="baseline",
    )
    result_dir = manager_result_dir(args, args.original_ckpt, run_tag)
    baseline.update(
        {
            "status": "success" if ok else "failed",
            "last_attempt_ts": utc_now(),
            "episodes": args.baseline_num_episodes,
            "num_inference_steps": args.num_inference_steps,
            "run_tag": run_tag,
            "result_dir": str(result_dir),
            "summary": str(result_dir / "summary.json"),
        }
    )
    write_json(args.state_path, state)


def should_retry(item: dict[str, Any], retry_seconds: int) -> bool:
    raw_ts = item.get("last_attempt_ts")
    if not raw_ts:
        return True
    try:
        last = datetime.strptime(str(raw_ts), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - last).total_seconds() >= retry_seconds


def run_checkpoint_eval(step: int, checkpoint_dir: Path, args: argparse.Namespace, state: dict[str, Any], recorder: Recorder) -> None:
    checkpoints = state.setdefault("checkpoints", {})
    entry = checkpoints.setdefault(str(step), {"checkpoint_dir": str(checkpoint_dir)})
    if (
        entry.get("status") == "success"
        and int(entry.get("episodes", -1)) == args.eval_num_episodes
        and int(entry.get("num_inference_steps", -1)) == args.num_inference_steps
        and not args.force
    ):
        return
    if not should_retry(entry, args.render_retry_seconds) and not args.force:
        return

    recorder.event("checkpoint_detected", step=step, checkpoint=str(checkpoint_dir))
    exported = export_checkpoint(step, checkpoint_dir, args, recorder)
    if exported is None:
        entry.update(
            {
                "status": "export_failed",
                "last_attempt_ts": utc_now(),
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
        write_json(args.state_path, state)
        return

    if args.probe_render and not probe_render(args, recorder):
        entry.update(
            {
                "status": "pending_render_unavailable",
                "last_attempt_ts": utc_now(),
                "checkpoint_dir": str(checkpoint_dir),
                "exported_checkpoint": str(exported),
            }
        )
        write_json(args.state_path, state)
        recorder.event("checkpoint_wait_render", step=step, retry_seconds=args.render_retry_seconds)
        return

    run_tag = (
        f"checkpoint_{step:09d}_{args.eval_num_episodes}trials_"
        f"{args.num_inference_steps}step_{compact_timestamp()}"
    )
    ok = run_robotwin_manager(
        ckpt=exported,
        dataset_stats=args.train_dataset_stats,
        episodes=args.eval_num_episodes,
        run_tag=run_tag,
        args=args,
        recorder=recorder,
        kind="checkpoint",
        step=step,
    )
    result_dir = manager_result_dir(args, exported, run_tag)
    entry.update(
        {
            "status": "success" if ok else "failed",
            "last_attempt_ts": utc_now(),
            "checkpoint_dir": str(checkpoint_dir),
            "exported_checkpoint": str(exported),
            "episodes": args.eval_num_episodes,
            "num_inference_steps": args.num_inference_steps,
            "run_tag": run_tag,
            "result_dir": str(result_dir),
            "summary": str(result_dir / "summary.json"),
        }
    )
    write_json(args.state_path, state)


def validate_paths(args: argparse.Namespace) -> None:
    required = [
        args.lightx2v_python,
        args.robotwin_python,
        args.fastwam_root,
        args.lightx2v_train_root,
        args.train_config,
        args.original_ckpt,
        args.original_dataset_stats,
        args.wan22_model_root,
        args.wan21_tokenizer_root,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(str(path))
    args.eval_root.mkdir(parents=True, exist_ok=True)
    args.export_dir.mkdir(parents=True, exist_ok=True)
    if not args.train_dataset_stats.exists() and args.train_run_dir.exists():
        fallback = args.train_run_dir / "dataset_stats.json"
        if fallback.exists():
            args.train_dataset_stats = fallback
    if not args.train_dataset_stats.exists():
        raise FileNotFoundError(str(args.train_dataset_stats))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Keep polling until interrupted.")
    parser.add_argument("--run-baseline", action="store_true", help="Run or retry original checkpoint baseline.")
    parser.add_argument("--baseline-only", action="store_true", help="Only run or retry the original baseline.")
    parser.add_argument("--probe-render-only", action="store_true", help="Run SAPIEN render probe and exit.")
    parser.add_argument("--force", action="store_true", help="Retry even if state says success or recently failed.")
    parser.add_argument("--force-once", action="store_true", help="Force the first polling pass, then resume normal retry rules.")
    parser.add_argument("--probe-render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=positive_int, default=60)
    parser.add_argument("--report-seconds", type=positive_int, default=1800)
    parser.add_argument("--render-retry-seconds", type=positive_int, default=1800)
    parser.add_argument("--ready-age-seconds", type=positive_int, default=60)
    parser.add_argument("--step-interval", type=positive_int, default=2000)
    parser.add_argument("--max-step", type=int, default=30000)
    parser.add_argument("--eval-num-episodes", type=positive_int, default=1)
    parser.add_argument("--baseline-num-episodes", type=positive_int, default=100)
    parser.add_argument("--num-inference-steps", type=positive_int, default=1)
    parser.add_argument("--num-gpus", type=positive_int, default=8)
    parser.add_argument("--max-tasks-per-gpu", type=positive_int, default=1)
    parser.add_argument("--replan-steps", type=positive_int, default=24)
    parser.add_argument("--task-name", default=None, help="Optional single RoboTwin task for debugging.")
    parser.add_argument("--lightx2v-root", type=expand, default=LIGHTX2V_ROOT)
    parser.add_argument("--lightx2v-train-root", type=expand, default=LIGHTX2V_TRAIN_ROOT)
    parser.add_argument("--fastwam-root", type=expand, default=FASTWAM_ROOT)
    parser.add_argument("--lightx2v-python", type=expand, default=LIGHTX2V_PYTHON)
    parser.add_argument("--robotwin-python", type=expand, default=ROBOTWIN_PYTHON)
    parser.add_argument("--train-run-dir", type=expand, default=DEFAULT_RUN_DIR)
    parser.add_argument("--train-config", type=expand, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--train-dataset-stats", type=expand, default=DEFAULT_RUN_DIR / "dataset_stats.json")
    parser.add_argument("--original-ckpt", type=expand, default=DEFAULT_ORIGINAL_CKPT)
    parser.add_argument("--original-dataset-stats", type=expand, default=DEFAULT_ORIGINAL_STATS)
    parser.add_argument("--eval-root", type=expand, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--export-dir", type=expand, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--wan22-model-root", type=expand, default=DEFAULT_WAN22_MODEL_ROOT)
    parser.add_argument("--wan21-tokenizer-root", type=expand, default=DEFAULT_WAN21_TOKENIZER_ROOT)
    parser.add_argument(
        "--vk-icd-filenames",
        type=str,
        default=None,
        help="Vulkan ICD manifest. Defaults to local NVIDIA fixup under eval-root, then SAPIEN bundled ICD.",
    )
    parser.add_argument(
        "--egl-vendor-library-filenames",
        type=str,
        default=None,
        help="GLVND EGL vendor manifest. Defaults to local NVIDIA fixup under eval-root when present.",
    )
    parser.add_argument("--glx-vendor-library-name", default="nvidia")
    parser.add_argument("--nvidia-driver-lib-dir", type=str, default=None)
    parser.add_argument("--vulkan-loader-lib-dir", type=str, default=None)
    parser.add_argument("--nvidia-visible-devices", default="all")
    args, manager_overrides = parser.parse_known_args()
    args.manager_overrides = manager_overrides
    args.records_path = args.eval_root / "eval_records.jsonl"
    args.state_path = args.eval_root / "state.json"
    args.monitor_log = args.eval_root / "monitor.log"
    return args


def main() -> int:
    args = parse_args()
    recorder = Recorder(args.records_path, args.monitor_log)
    validate_paths(args)
    if args.probe_render_only:
        return 0 if probe_render(args, recorder) else 2

    state = read_json(args.state_path, {"baseline": {}, "checkpoints": {}})
    persistent_force = args.force
    if args.force_once:
        args.force = True
    recorder.event(
        "monitor_start",
        watch=args.watch,
        force=args.force,
        force_once=args.force_once,
        train_run_dir=str(args.train_run_dir),
        eval_root=str(args.eval_root),
        step_interval=args.step_interval,
        eval_num_episodes=args.eval_num_episodes,
        baseline_num_episodes=args.baseline_num_episodes,
        num_inference_steps=args.num_inference_steps,
        manager_overrides=args.manager_overrides,
    )

    last_report = 0.0
    while True:
        if args.run_baseline or args.baseline_only:
            run_baseline(args, state, recorder)

        if not args.baseline_only:
            for step, checkpoint_dir in list_target_checkpoints(args):
                run_checkpoint_eval(step, checkpoint_dir, args, state, recorder)

        if args.force_once and not persistent_force:
            args.force = False
            args.force_once = False
            recorder.event("force_once_consumed")

        now = time.time()
        if now - last_report >= args.report_seconds:
            progress = parse_train_progress(args.train_run_dir / "train.log")
            done = sorted(
                int(step)
                for step, item in state.get("checkpoints", {}).items()
                if isinstance(item, dict) and item.get("status") == "success"
            )
            recorder.event(
                "monitor_report",
                train_progress=progress,
                successful_checkpoint_steps=done,
                baseline_status=state.get("baseline", {}).get("status"),
            )
            last_report = now

        if not args.watch:
            break
        time.sleep(args.poll_seconds)

    recorder.event("monitor_exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
