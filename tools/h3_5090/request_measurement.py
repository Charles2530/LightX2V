"""Opt-in H3 request observation, without changing the scheduler equations."""

import subprocess
import time
import xml.etree.ElementTree as ET

import torch


def query_gpu_boundary(timeout=5):
    """One bounded driver call. A failed observation is not a conflict."""
    started = time.monotonic()
    try:
        data = subprocess.check_output(["nvidia-smi", "-q", "-x"], timeout=timeout)
        devices = []
        for gpu in ET.fromstring(data).findall("gpu"):
            devices.append(
                {
                    "uuid": gpu.findtext("uuid"),
                    "memory_mib": gpu.findtext("fb_memory_usage/used"),
                    "utilization_pct": gpu.findtext("utilization/gpu_util"),
                    "processes": [{"pid": int(p.findtext("pid")), "type": p.findtext("type")} for p in gpu.findall("processes/process_info")],
                }
            )
        return {"status": "observed", "devices": devices, "elapsed_s": time.monotonic() - started}
    except subprocess.TimeoutExpired:
        return {"status": "resource_query_timeout", "elapsed_s": time.monotonic() - started}
    except (OSError, subprocess.CalledProcessError, ET.ParseError, ValueError) as error:
        return {"status": "resource_query_error", "error": type(error).__name__, "elapsed_s": time.monotonic() - started}


def boundary_verdict(before, after, task_pids):
    """No inference of continuous exclusivity from two boundary observations."""
    observed = all(x["status"] == "observed" for x in (before, after))
    foreign = sorted({p["pid"] for point in (before, after) for gpu in point.get("devices", []) for p in gpu["processes"] if p["pid"] not in task_pids})
    return {
        "protocol": "request_boundaries_only",
        "observations_complete": observed,
        "foreign_pids": foreign,
        "boundary_conflict": bool(foreign),
        "boundary_checks_pass": observed and not foreign,
        "resource_query_timeout": any(x["status"] == "resource_query_timeout" for x in (before, after)),
        "continuous_exclusivity_verified": False,
    }


class H3RequestObservation:
    """CUDA event intervals and exact final-latent capture for an existing runner.

    Events span the real inference call, including stream waits and CPU launch
    gaps; they are completion latency, not GPU kernel occupancy. Latent clones
    happen after the segment, are charged to request latency and copied to CPU
    only after generate returns. Call detach before discarding the runner.
    """

    def __init__(self, runner):
        self.runner = runner
        self.enabled = True
        self.steps = []
        self.finals = None
        self.audio_trajectory = []
        self._infer = runner.model.infer
        self._segment = runner.run_segment
        self.capture_audio = False
        runner.model.infer = self.infer
        runner.run_segment = self.segment

    def reset(self, *, enabled=True, audio=False):
        self.enabled = enabled
        self.capture_audio = audio
        self.steps = []
        self.finals = None
        self.audio_trajectory = []

    def infer(self, inputs):
        if not self.enabled:
            return self._infer(inputs)
        scheduler = self.runner.scheduler
        layout = scheduler.layout_cpu
        prefix = layout.token_tags[layout.text_indices]
        row = {
            "step": scheduler.step_index,
            "stage": "low" if getattr(scheduler, "full_state", None) is not None and not scheduler.lifted else "high",
            "latent_hw": [scheduler.latent_height, scheduler.latent_width],
            "qwen_prefix_rows": prefix.numel(),
            "prefix_text_rows": int((prefix == 1).sum()),
            "prefix_visual_rows": int((prefix == 0).sum()),
            "prefix_audio_rows": int((prefix == 2).sum()),
            "condition_video_rows": layout.num_condition_video_rows,
            "target_video_rows": scheduler.video_latents.shape[0] - layout.num_condition_video_rows,
            "audio_rows": scheduler.audio_latents.shape[0],
            "packed_rows": layout.sequence_length,
            "video_sigma": float(scheduler.video_sigmas[scheduler.step_index]),
            "audio_sigma": float(scheduler.audio_sigmas[scheduler.step_index]),
        }
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        result = self._infer(inputs)
        manager = getattr(self.runner.model.transformer_infer, "offload_manager", None)
        if manager is not None:
            torch.cuda.current_stream().wait_stream(manager.compute_stream)
            torch.cuda.current_stream().wait_stream(manager.cuda_load_stream)
        end.record()
        row.update(start_event=start, end_event=end, host_infer_s=time.perf_counter() - wall_start)
        self.steps.append(row)
        # Pre-update audio is useful later for trajectory coupling diagnostics.
        if self.capture_audio:
            self.audio_trajectory.append(scheduler.audio_latents.detach().clone())
        return result

    def segment(self, *args, **kwargs):
        result = self._segment(*args, **kwargs)
        self.finals = tuple(x.detach().clone() for x in result)
        if self.capture_audio:
            self.audio_trajectory.append(result[1].detach().clone())
        return result

    def export(self):
        torch.cuda.synchronize()
        rows = []
        for source in self.steps:
            row = {k: v for k, v in source.items() if k not in ("start_event", "end_event")}
            row["dit_cuda_event_ms"] = source["start_event"].elapsed_time(source["end_event"])
            rows.append(row)
        return rows

    def detach(self):
        self.runner.model.infer = self._infer
        self.runner.run_segment = self._segment
