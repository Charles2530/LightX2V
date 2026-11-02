r"""Measure H3 forward scheduling with real weights; no repository patches.

Examples (run with the repository's Python environment):
  python bench_ref.py --task t2av --model-path /models/MiniMax-H3 \
      --config minimax_h3_profile_runtime.json --output /tmp/h3-t2av.json
  python bench_ref.py --task refa2v --model-path /models/MiniMax-H3 \
      --dit-checkpoint /models/causal/gen-base_dit.pt \
      --image assets/inputs/audio/seko_input.png \
      --audio assets/inputs/audio/seko_input.mp3 --output /tmp/h3-causal.json

Single CUDA device only: distributed VAE/NCCL overlap needs a separate design.
Recent torchaudio versions need a matching TorchCodec build for driving audio.
All variants use identical weights, schedules, seeds and resident VAEs. Baseline
calls the original runner. Causal variants overlap RGB decoding and/or prefetch
the next *truncated prefix* of audio, preserving the encoder's boundary context.
Input encoding, DiT preparation/offload, RGB delivery and final audio decoding
are timed; loading, warmup, correctness checks and MP4 encoding are excluded.
The default small causal canvas fits the unsharded model plus step KV cache.
Pass --size 544 960 explicitly to measure the resolution in the original log.

Additional causal probes: scheduler_cache preloads per-request layouts/noise/
timesteps; video_scheduler combines that with RGB overlap. video_head skips the
unused causal audio output head. unpatchify_async measures moving latent
unpatchifying onto the decode stream; forward_async combines that with the
other forward-safe probes so the main stream can start the next DiT chunk sooner.
early_audio encodes
the exact audio prefixes during condition prefill. fast_offload reuses immutable
pinned CPU weights instead of copying DiT weights back from GPU. hidden_reuse
reuses a per-layout DiT input buffer instead of cloning the full sequence every
step. optimized combines these changes with video overlap. Optional resident probes keep
DiT on the GPU throughout requests. Their setup/teardown are reported outside
timing; their gains do NOT apply to cold starts or low-memory GPUs. OOM probes
are reported as skipped, never treated as successful or silently resized.
condition_cache_reuse is an explicit experimental probe for reusing the fixed
causal condition KV cache. It is excluded from optimized because the current
checkpoint's per-step condition K/V are not identical and the correctness gate
will reject the probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

CAUSAL_MODES = {
    "baseline": frozenset(),
    "video_overlap": frozenset(("video",)),
    "audio_prefetch": frozenset(("audio",)),
    "combined": frozenset(("video", "audio")),
    "scheduler_cache": frozenset(("schedule",)),
    "video_scheduler": frozenset(("video", "schedule")),
    "early_audio": frozenset(("early_audio",)),
    "fast_offload": frozenset(("fast_offload",)),
    "hidden_reuse": frozenset(("hidden_reuse",)),
    "condition_cache_reuse": frozenset(("condition_cache_reuse",)),
    "video_head": frozenset(("video_head",)),
    "unpatchify_async": frozenset(("video", "unpatchify")),
    "forward_async": frozenset(("hidden_reuse", "video_head", "video", "schedule", "unpatchify")),
    "inference_optimized": frozenset(("hidden_reuse", "video", "schedule")),
    "optimized": frozenset(("fast_offload", "hidden_reuse", "video_head", "video", "schedule", "early_audio", "unpatchify")),
    "resident": frozenset(("resident",)),
    "resident_video_scheduler": frozenset(("resident", "video", "schedule")),
    "resident_optimized": frozenset(("resident", "video", "schedule", "early_audio")),
}

# These must precede LightX2V imports. Profiling barriers destroy overlap.
sys.dont_write_bytecode = True
os.environ["PROFILING_DEBUG_LEVEL"] = "0"
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "FP32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("t2av", "ref2av", "refa2v"), default="refa2v")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", help="Optional existing JSON; overrides below stay in memory")
    parser.add_argument("--dit-checkpoint", help="Causal full checkpoint file or safetensors directory; not a non-causal Turbo LoRA")
    parser.add_argument("--base-causal-proxy", action="store_true", help="Explicitly label a causal scheduling probe using non-causal base weights; NOT a causal quality benchmark")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/bench_ref_adaln"), help="Build/reuse checkpoint-specific AdaLN caches outside the repository")
    parser.add_argument("--image", default="assets/inputs/audio/seko_input.png")
    parser.add_argument("--audio", default="assets/inputs/audio/seko_input.mp3")
    parser.add_argument("--prompt", default="A person speaks naturally, with synchronized lip movements and subtle facial expressions. The camera remains stationary.")
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA device index")
    parser.add_argument("--offload-granularity", choices=("model", "block"), help="Override DiT CPU offload granularity; block uses two GPU staging buffers")
    parser.add_argument("--attn-type", help="Override DiT attention backend, e.g. torch_sdpa or sage_attn2")
    parser.add_argument("--rms-type", help="Override RMSNorm backend")
    parser.add_argument("--rope-type", help="Override RoPE backend")
    parser.add_argument("--use-compile", action="store_true", help="Compile each DiT block with torch.compile; compile/warmup time is excluded from measured requests")
    parser.add_argument("--compile-backend", help="torch.compile backend override, for example default or a registered platform backend")
    parser.add_argument("--use-fused-qkv", action="store_true", help="Use the H3 fused QKV projection when the backend and weights support it")
    parser.add_argument("--use-fused-qkv-norm-rope", action="store_true", help="Fuse QKV projection postprocessing, Q/K RMSNorm and RoPE when supported")
    parser.add_argument("--qkv-norm-rope-type", help="Fused QKV Norm/RoPE backend, usually triton")
    parser.add_argument("--dit-prepost-resident", action="store_true", help="With block offload, keep DiT pre/post weights on GPU")
    parser.add_argument("--dit-release-block-offload-buffers", action="store_true", help="Release block staging buffers between requests")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests per variant")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", help=f"refa2v: {' '.join(CAUSAL_MODES)}; t2av: baseline vae_overlap. Resident probes require additional GPU headroom.")
    parser.add_argument("--queue-depth", type=int, default=2, help="Bound pending RGB copies/callbacks")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, help="Optional JSON report (use /tmp to keep the repository clean)")
    args = parser.parse_args()
    allowed = tuple(CAUSAL_MODES) if args.task == "refa2v" else ("baseline", "vae_overlap")
    args.modes = args.modes or (
        ["baseline", "hidden_reuse", "video_head", "unpatchify_async", "forward_async", "inference_optimized", "fast_offload", "optimized"] if args.task == "refa2v" else list(allowed)
    )
    if not args.modes or args.modes[0] != "baseline" or len(set(args.modes)) != len(args.modes) or set(args.modes) - set(allowed):
        parser.error(f"--modes must start with baseline, be unique, and belong to {allowed}")
    if args.repeats < 1 or args.warmup < 0 or args.queue_depth < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("Invalid repetitions, queue depth or tolerances")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This benchmark supports one CUDA process; do not launch with multi-rank torchrun")
    if args.base_causal_proxy and args.task != "refa2v":
        parser.error("--base-causal-proxy is only meaningful for refa2v")
    return args


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def install_audio_load_fallback(torchaudio):
    """Keep the benchmark runnable when torchaudio's TorchCodec build is absent.

    Recent torchaudio releases route ``load`` exclusively through TorchCodec.
    The repository only needs a float waveform for the H3 input encoder, so a
    PyAV fallback is sufficient for this benchmark and leaves torchaudio's
    native path untouched when it works.
    """
    if getattr(torchaudio.load, "_bench_ref_fallback", False):
        return
    native_load = torchaudio.load

    def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
        try:
            return native_load(uri, frame_offset=frame_offset, num_frames=num_frames, normalize=normalize, channels_first=channels_first, format=format)
        except (ImportError, OSError, RuntimeError):
            import av
            import numpy as np
            import torch

            container = av.open(uri)
            stream = next(item for item in container.streams if item.type == "audio")
            chunks = [torch.from_numpy(frame.to_ndarray()) for frame in container.decode(stream)]
            if not chunks:
                return torch.empty((stream.codec_context.channels, 0), dtype=torch.float32), int(stream.rate)
            waveform = torch.cat(chunks, dim=1)
            begin = max(int(frame_offset), 0)
            end = None if num_frames is None or int(num_frames) < 0 else begin + int(num_frames)
            waveform = waveform[:, begin:end]
            if not channels_first:
                waveform = waveform.transpose(0, 1)
            # PyAV may expose integer PCM; torchaudio's normalized contract is
            # float32 in [-1, 1]. Avoid a dtype-dependent division for floats.
            if torch.is_floating_point(waveform):
                waveform = waveform.float()
            elif normalize:
                info = np.iinfo(waveform.numpy().dtype)
                waveform = waveform.float() / max(abs(info.min), info.max)
            else:
                waveform = waveform.float()
            return waveform, int(stream.rate)

    load._bench_ref_fallback = True
    torchaudio.load = load


@contextmanager
def replace_attribute(obj, name, value):
    """Restore even instance-vs-class attribute lookup on exceptions."""
    existed = name in obj.__dict__
    previous = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


class Trace:
    def __init__(self, torch):
        self.torch = torch
        self.events = []
        self.host_stage_ms = defaultdict(float)
        self.deliveries = []
        self.start = time.perf_counter()
        self.segment_start = None
        self.segment_end = None
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record()

    @contextmanager
    def stage(self, name):
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))

    def wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            begin = time.perf_counter()
            try:
                with self.stage(name):
                    return fn(*args, **kwargs)
            finally:
                self.host_stage_ms[name] += (time.perf_counter() - begin) * 1000

        return wrapped

    def callback(self, frames, start_frame, is_final):
        self.deliveries.append(
            {
                "start_frame": start_frame,
                "frames": frames.shape[2],
                "is_final": is_final,
                "ready_ms": (time.perf_counter() - self.start) * 1000,
            }
        )

    def finish(self, wall_ms):
        stages = defaultdict(float)
        timeline = []
        for name, start, end in self.events:
            duration = start.elapsed_time(end)
            stages[name] += duration
            timeline.append({"stage": name, "start_ms": self.origin.elapsed_time(start), "duration_ms": duration})
        first = self.deliveries[0]["ready_ms"] if self.deliveries else None
        gaps = [b["ready_ms"] - a["ready_ms"] for a, b in zip(self.deliveries, self.deliveries[1:]) if not b["is_final"]]
        return {
            "pipeline_ms": wall_ms,
            "forward_ms": None if self.segment_start is None else wall_ms - (self.segment_start - self.start) * 1000,
            "segment_ms": None if self.segment_end is None else (self.segment_end - self.segment_start) * 1000,
            "first_rgb_ms": first,
            "first_rgb_from_segment_ms": None if first is None or self.segment_start is None else first - (self.segment_start - self.start) * 1000,
            "chunk_gap_p50_ms": percentile(gaps, 0.5),
            "chunk_gap_p95_ms": percentile(gaps, 0.95),
            "chunk_gaps_ms": gaps,
            "deliveries": self.deliveries,
            "peak_allocated_gib": self.torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": self.torch.cuda.max_memory_reserved() / 1024**3,
            "stage_event_ms": dict(stages),
            "host_stage_ms": dict(self.host_stage_ms),
            "timeline": timeline,
        }


class RGBDelivery:
    """Ordered CPU delivery without blocking the DiT submission thread.

    CUDA copies are enqueued on the decode stream; the consumer waits only on
    that copy's event. Holding source references prevents allocator reuse.
    """

    def __init__(self, torch, callback, depth):
        self.torch, self.callback, self.depth = torch, callback, depth
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-rgb")
        self.pending = deque()
        self.outputs = []

    def submit(self, frames, start_frame, final):
        host = self.torch.empty(frames.shape, dtype=frames.dtype, device="cpu", pin_memory=True)
        host.copy_(frames, non_blocking=True)
        done = self.torch.cuda.Event()
        done.record()

        def consume(source=frames):
            # Keep the source alive until its nonblocking copy is complete.
            done.synchronize()
            if self.callback is not None:
                self.callback(host, start_frame, final)
            return host

        self.pending.append(self.pool.submit(consume))
        if len(self.pending) > self.depth:
            self.outputs.append(self.pending.popleft().result())

    def finish(self):
        try:
            while self.pending:
                self.outputs.append(self.pending.popleft().result())
            return self.outputs
        finally:
            self.pool.shutdown(wait=True)


@contextmanager
def cache_causal_schedule(scheduler):
    """Per-request metadata only; no reuse of prompts, outputs or KV across calls.

    Each native step_pre performs synchronous small H2D copies. Prepare the
    exact same tensors once, then select them without draining the compute
    stream. Chunk noise is also uploaded once, preserving RNG draws and dtype.
    """
    from contextlib import ExitStack

    import torch

    from lightx2v.models.networks.minimax_h3.packing import build_row_timesteps
    from lightx2v.models.schedulers.minimax_h3.scheduler import _layout_to_device

    original_prepare = scheduler.prepare
    original_condition = scheduler.prepare_condition
    entries = {}
    noise = None

    def prepare(*args, **kwargs):
        nonlocal noise
        entries.clear()
        original_prepare(*args, **kwargs)
        device = scheduler.condition_video.device
        noise = scheduler.video_noise.to(device, non_blocking=True)
        layouts = [("condition", scheduler.plan.condition_layout)] + [(i, scheduler.plan.chunk_layout(i)) for i in range(len(scheduler.plan.chunks))]
        previous_end = scheduler.plan.condition_layout.sequence_length
        for key, layout in layouts:
            gpu_layout = _layout_to_device(layout, device)
            if key == "condition":
                positions, start, evicted = gpu_layout.position_ids, 0, 0
            else:
                prefix = scheduler.plan.prefix_positions(key)
                positions = torch.cat((prefix, layout.position_ids)).to(device, non_blocking=True)
                start = prefix.shape[0]
                evicted = previous_end - start
                previous_end = start + layout.sequence_length
            steps = []
            for timestep in scheduler.video_timesteps_cpu:
                unique, inverse = build_row_timesteps(layout, timestep, 1.0, condition_video_timestep=0.999)
                steps.append((unique, unique.to(device, non_blocking=True), inverse.to(device, non_blocking=True)))
            entries[key] = (layout, gpu_layout, positions, start, evicted, steps)

    def select(key):
        layout, gpu_layout, positions, start, evicted, _ = entries[key]
        scheduler.prepared_inputs = None
        scheduler.layout_cpu, scheduler.layout = layout, gpu_layout
        scheduler.key_position_ids = positions
        scheduler.cache_start, scheduler.evicted_rows = start, evicted

    def condition(*, overwrite=False):
        if not entries:
            return original_condition(overwrite=overwrite)
        select("condition")
        scheduler.overwrite_condition = overwrite
        scheduler.condition_phase = True
        scheduler.video_latents = scheduler.condition_video
        scheduler.audio_latents = scheduler.condition_video.new_empty((0, 32))

    def chunk(index, clean_audio):
        select(index)
        scheduler.overwrite_condition = False
        scheduler.condition_phase = False
        scheduler.chunk_index = index
        scheduler.video_latents = noise[scheduler.plan.chunks[index].video_rows].clone()
        scheduler.audio_latents = clean_audio.to(noise.device)

    def step(index):
        scheduler.step_index = int(index)
        scheduler.cache.current_step = scheduler.step_index
        key = "condition" if scheduler.condition_phase else scheduler.chunk_index
        scheduler.unique_timesteps_cpu, scheduler.unique_timesteps, scheduler.timestep_indices = entries[key][-1][scheduler.step_index]

    with ExitStack() as stack:
        for name, method in (("prepare", prepare), ("prepare_condition", condition), ("prepare_chunk", chunk), ("step_pre", step)):
            stack.enter_context(replace_attribute(scheduler, name, method))
        try:
            yield
        finally:
            entries.clear()
            noise = None


@contextmanager
def immutable_dit_offload(runner):
    """Release GPU aliases of unchanged weights, keeping existing pinned sources.

    This is already how the native Qwen text backbone offloads its immutable
    weights. Restrict the experiment to unquantized, unfused, non-LoRA model
    offload so there are no mutable weights or fused storage aliases to repair.
    Other buffers retain their original offload implementation.
    """
    from contextlib import ExitStack

    import torch

    cfg = runner.config
    if runner.model.block_offload or any(cfg.get(key, False) for key in ("use_fused_qkv", "dit_quantized", "lora_dynamic_apply", "lora_configs", "lora_path")):
        raise ValueError("fast_offload requires unquantized, unfused, immutable model-level weights without LoRA")
    leaves, seen = [], set()
    for container in (runner.model.pre_weight, runner.model.transformer_weights, runner.model.post_weight):
        for _, leaf in container.named_weight_leaves():
            if id(leaf) in seen or not getattr(leaf, "base_attrs", None) or not hasattr(leaf, "to_cpu"):
                continue
            seen.add(id(leaf))
            sources = {}
            for _, attr, _ in leaf.base_attrs:
                source = getattr(leaf, "pin_" + attr, None)
                active = getattr(leaf, attr, None)
                if source is None and active is None:
                    continue
                if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
                    raise ValueError(f"fast_offload requires an existing CPU source for {type(leaf).__name__}.{attr}")
                if active is not None and (active.shape != source.shape or active.dtype != source.dtype):
                    raise ValueError(f"CPU source does not match active weight: {type(leaf).__name__}.{attr}")
                sources[attr] = source
            if any(getattr(leaf, attr, None) is not None for attr in getattr(leaf, "lora_attrs", {})):
                # Native RMSNorm keeps a scalar zero weight_diff even without
                # LoRA. Leave any auxiliary/diff-bearing operator on its native
                # offload path; the large plain linear weights remain eligible.
                continue
            leaves.append((leaf, sources))
    if not leaves:
        raise ValueError("No immutable DiT weight leaves found")

    def release(sources):
        def to_cpu(self, non_blocking=False):
            for attr, tensor in sources.items():
                setattr(self, attr, tensor)

        return to_cpu

    with ExitStack() as stack:
        for leaf, sources in leaves:
            stack.enter_context(replace_attribute(leaf, "to_cpu", MethodType(release(sources), leaf)))
        yield {"immutable_weight_leaves": len(leaves), "cpu_source_bytes": sum(t.numel() * t.element_size() for _, sources in leaves for t in sources.values())}


@contextmanager
def reuse_causal_hidden_buffer(runner):
    """Avoid cloning the complete packed hidden-state sequence every DiT step.

    Causal H3 changes only the video rows between steps; text/audio rows and
    rotary embeddings stay fixed for one layout. The native path clones the
    whole template before replacing those video rows. This probe keeps one
    private mutable buffer per active layout and updates only video rows. The
    transformer consumes the buffer functionally, so it is safe to reuse after
    each infer call; a new layout/chunk naturally creates a new buffer.
    """
    if not hasattr(runner.model, "pre_infer") or not runner.model.__class__.__name__.endswith("CausalModel"):
        raise ValueError("hidden_reuse is only supported by the causal H3 model")
    pre_infer = runner.model.pre_infer
    original = pre_infer.prepare_inputs
    working = {"key": None, "hidden": None}

    def prepare(self, weights, prompt_embeds):
        scheduler = self.scheduler
        if scheduler.prepared_inputs is None:
            hidden_states, rotary_emb = original(weights, prompt_embeds)
            # original() has populated scheduler.prepared_inputs and returned a
            # clone suitable for the first inference call. Keep that clone as
            # the reusable mutable buffer for this layout.
            working["key"] = id(scheduler.prepared_inputs[0])
            working["hidden"] = hidden_states
            return hidden_states, rotary_emb
        key = id(scheduler.prepared_inputs[0])
        if key != working["key"]:
            raise RuntimeError("causal prepared-input layout changed without rebuilding the reusable buffer")
        hidden_states, rotary_emb, _ = scheduler.prepared_inputs
        if not scheduler.condition_phase:
            video_embeds = weights.proj_in.apply(scheduler.video_latents.float()).to(hidden_states.dtype)
            working["hidden"].index_copy_(0, scheduler.layout.video_indices, video_embeds)
        return working["hidden"], rotary_emb

    with replace_attribute(pre_infer, "prepare_inputs", MethodType(prepare, pre_infer)):
        try:
            yield {"hidden_buffer_reuse": True}
        finally:
            working["key"] = None
            working["hidden"] = None


@contextmanager
def reuse_causal_condition_cache(runner):
    """Build the fixed 0.999 condition KV cache once and replicate its steps.

    Causal H3 uses a step-isolated KV pool because media denoising steps have
    different latents. The condition pass is different: its reference/video
    rows and timestep are fixed at 0.999 for every step. Re-running the full
    DiT four times therefore only repopulates identical condition K/V rows.
    This probe runs step zero once and copies its physical K/V buffers to the
    remaining step slots. It is restricted to the native in-GPU step cache;
    unsupported cache implementations fail explicitly.
    """
    if not hasattr(runner.model, "pre_infer") or not runner.model.__class__.__name__.endswith("CausalModel"):
        raise ValueError("condition_cache_reuse is only supported by the causal H3 model")

    def fill(self, *, overwrite=False):
        scheduler = self.scheduler
        cache = scheduler.cache
        if not hasattr(cache, "_k_buffer") or not hasattr(cache, "_v_buffer") or not hasattr(cache, "num_steps"):
            raise ValueError("condition_cache_reuse requires the native in-GPU StepRollingKVCachePool")
        scheduler.prepare_condition(overwrite=overwrite)
        self.check_stop()
        scheduler.step_pre(0)
        self.model.infer(self.inputs)
        if int(cache.num_steps) > 1:
            # The first dimension is [step, layer, token, head, dim]. The
            # condition pass also records the logical ends for every layer.
            cache._k_buffer[1:].copy_(cache._k_buffer[0])
            cache._v_buffer[1:].copy_(cache._v_buffer[0])
            cache._global_end[1:].copy_(cache._global_end[0])
            cache._local_end[1:].copy_(cache._local_end[0])
            cache._ring_active[1:].copy_(cache._ring_active[0])
            cache._ring_sink[1:].copy_(cache._ring_sink[0])
            cache._ring_head[1:].copy_(cache._ring_head[0])
            cache._ring_recent_len[1:].copy_(cache._ring_recent_len[0])

    with replace_attribute(runner, "fill_condition_cache", MethodType(fill, runner)):
        yield {"condition_cache_reuse": True, "condition_steps_built": 1}


@contextmanager
def causal_video_only_head(runner):
    """Skip the unused causal audio output head without changing video math.

    Causal H3 treats the driving-audio latent as fixed conditioning and its
    scheduler never consumes ``audio_noise_pred``. The native post pass still
    projects every packed row through both output heads. Keep the exact final
    norm/modulation and video projection order, and omit only the unused
    ``audio_proj_out``. The baseline path is untouched and correctness is
    checked by the benchmark's output comparison.
    """
    if not hasattr(runner.model, "post_infer") or not runner.model.__class__.__name__.endswith("CausalModel"):
        raise ValueError("video_head is only supported by the causal H3 model")

    import torch.nn.functional as F

    from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3VelocityOutput
    from lightx2v.utils.envs import GET_DTYPE

    post_infer = runner.model.post_infer

    def infer(self, weights, hidden_states, pre_infer_out):
        modulation = pre_infer_out.norm_out_modulation
        if modulation is None:
            if pre_infer_out.temb is None:
                raise RuntimeError("MiniMax-H3 final-norm modulation is missing")
            modulation = weights.norm_out_linear.apply(F.silu(pre_infer_out.temb).to(GET_DTYPE()))
        shift, scale = modulation.chunk(2, dim=-1)
        indices = pre_infer_out.timestep_indices
        hidden_states = weights.norm_out.apply(hidden_states)
        hidden_states = hidden_states * (1.0 + scale.index_select(0, indices))
        hidden_states = hidden_states + shift.index_select(0, indices)
        # Keep the full video projection before selecting rows. Although a
        # row-subset GEMM is mathematically equivalent, changing the GEMM
        # shape can change reduction order and break strict output parity.
        video = weights.proj_out.apply(hidden_states.float()).index_select(0, pre_infer_out.video_indices)
        return MiniMaxH3VelocityOutput(video=video, audio=None)

    with replace_attribute(post_infer, "infer", MethodType(infer, post_infer)):
        yield {"video_only_head": True}


@contextmanager
def resident_dit(runner, enabled):
    """Explicit warm-service policy; exclude and report setup/cleanup costs."""
    import torch

    stats = {"resident_setup_ms": 0.0, "resident_cleanup_ms": 0.0}
    if not enabled:
        yield stats
        return
    if runner.model.block_offload:
        raise ValueError("Resident probes require model-level offload, not block offload")
    original = runner.config["cpu_offload"]
    begin = time.perf_counter()
    try:
        runner.model.to_cuda()
        torch.cuda.synchronize()
        stats["resident_setup_ms"] = (time.perf_counter() - begin) * 1000
        # Startup config is a locked mapping shared by several components.
        # Override the runner's policy locally without mutating that mapping.
        with replace_attribute(runner, "config", dict(runner.config, cpu_offload=False)):
            yield stats
    finally:
        if original:
            begin = time.perf_counter()
            runner.model.to_cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            stats["resident_cleanup_ms"] = (time.perf_counter() - begin) * 1000


def causal_segment(runner, trace, features, depth):
    import torch

    from lightx2v.models.networks.minimax_h3.packing import unpatchify_video_tokens
    from lightx2v.models.video_encoders.hf.minimax_h3_causal.streaming import MiniMaxH3StreamingVideoDecoder

    video_async = "video" in features
    audio_async = "audio" in features or "early_audio" in features
    runner.set_vae_decode_tile_shape()
    decoder = MiniMaxH3StreamingVideoDecoder(runner.video_vae, runner.request_num_frames)
    main_stream = torch.cuda.current_stream()
    vae_stream = torch.cuda.Stream() if video_async else main_stream
    audio_stream = torch.cuda.Stream() if audio_async else main_stream
    vae_stream.wait_stream(main_stream)
    audio_stream.wait_stream(main_stream)
    delivery = RGBDelivery(torch, runner.stream_callback, depth) if video_async else None
    if video_async:
        # Preserve the original postprocessing arithmetic, defer only .cpu().
        decoder.postprocess = lambda video: decoder.vae.postprocess(video).float()
    waveform = runner.waveform[:1] if runner.config.get("audio_mono", False) else runner.waveform
    if audio_async:
        # The file is already available. Only encode a truncated prefix, never
        # encode the full waveform then slice (that would leak future context).
        with torch.cuda.stream(audio_stream):
            waveform = waveform.to("cuda")
    chunks = runner.scheduler.plan.chunks
    videos, left, right = [], [], []
    emitted = 0

    def encode(index):
        chunk = chunks[index]
        with torch.cuda.stream(audio_stream):
            latents = runner.audio_vae.encode(waveform[..., : chunk.audio_frames.stop * runner.audio_vae.hop_length], return_cpu=False)
            if waveform.shape[0] == 1:
                latents = latents.expand(2, -1, -1)
            clean = latents[..., chunk.audio_frames].transpose(1, 2).reshape(-1, latents.shape[1]).contiguous()
            ready = torch.cuda.Event()
            ready.record()
        return clean, ready

    try:
        # Known-file mode only: every prefix retains its original right edge.
        # Work and preparation are charged to this request, never cached from
        # warmup. The condition DiT has no dependency on driving-audio latents.
        early = [encode(i) for i in range(len(chunks))] if "early_audio" in features else None
        runner.fill_condition_cache()
        prefetched = encode(0) if audio_async and early is None else None
        for index, chunk in enumerate(chunks):
            runner.check_stop()
            clean, ready = early[index] if early is not None else prefetched if audio_async else encode(index)
            main_stream.wait_event(ready)
            clean.record_stream(main_stream)
            # Audio(k+1) is independent of DiT(k); same prefix boundary as baseline.
            if audio_async and early is None:
                prefetched = encode(index + 1) if index + 1 < len(chunks) else None
            runner.scheduler.prepare_chunk(index, clean)
            for step in range(runner.scheduler.infer_steps):
                runner.check_stop()
                runner.scheduler.step_pre(step)
                runner.model.infer(runner.inputs)
                runner.scheduler.step_post()
                if runner.progress_callback:
                    runner.progress_callback(100 * (index * runner.scheduler.infer_steps + step + 1) / (len(chunks) * runner.scheduler.infer_steps), 100)
            # ``unpatchify`` only reads the completed chunk. In the async
            # probe, move it to the decode stream together with VAE streaming
            # so the main stream can prepare and denoise the next chunk. Keep
            # an explicit source reference until that stream has consumed it;
            # prepare_chunk() replaces scheduler.video_latents for the next
            # chunk but must not recycle this storage early.
            video_rows = runner.scheduler.video_latents
            if "unpatchify" in features:
                complete = torch.cuda.Event()
                complete.record(main_stream)
                with torch.cuda.stream(vae_stream):
                    vae_stream.wait_event(complete)
                    video_rows.record_stream(vae_stream)
                    with trace.stage("video_unpatchify"):
                        video = unpatchify_video_tokens(
                            video_rows,
                            chunk.video_frames.stop - chunk.video_frames.start,
                            runner.scheduler.latent_height,
                            runner.scheduler.latent_width,
                        )
                    for frames in decoder.push(video):
                        if delivery is not None:
                            delivery.submit(frames, emitted, False)
                        else:
                            videos.append(frames)
                            runner.stream_callback(frames, emitted, False)
                        emitted += frames.shape[2]
            else:
                video = unpatchify_video_tokens(
                    video_rows,
                    chunk.video_frames.stop - chunk.video_frames.start,
                    runner.scheduler.latent_height,
                    runner.scheduler.latent_width,
                )
                complete = torch.cuda.Event()
                complete.record(main_stream)
                with torch.cuda.stream(vae_stream):
                    vae_stream.wait_event(complete)
                    video.record_stream(vae_stream)
                    for frames in decoder.push(video):
                        if delivery is not None:
                            delivery.submit(frames, emitted, False)
                        else:
                            videos.append(frames)
                            runner.stream_callback(frames, emitted, False)
                        emitted += frames.shape[2]
            # Keep small clean latents on the GPU until the end. Baseline's
            # per-chunk .cpu() would wait for the just-submitted DiT.
            left.append(clean[: chunk.num_audio_frames])
            right.append(clean[chunk.num_audio_frames :])
        with torch.cuda.stream(vae_stream):
            tail = decoder.finish()
            if tail is not None:
                if delivery is not None:
                    delivery.submit(tail, emitted, True)
                else:
                    videos.append(tail)
                    runner.stream_callback(tail, emitted, True)
        if delivery is not None:
            videos = delivery.finish()
        audio = torch.cat((torch.cat(left), torch.cat(right))).cpu()
        return torch.cat(videos, dim=2), audio
    finally:
        # Also drain on exceptions, before the enclosing runner offloads DiT.
        vae_stream.synchronize()
        audio_stream.synchronize()
        if delivery is not None:
            delivery.finish()


def parallel_vae_decode(runner, video_rows, audio_rows):
    import torch

    from lightx2v.models.networks.minimax_h3.packing import unpack_audio_tokens, unpatchify_video_tokens

    video_rows = video_rows[runner.scheduler.num_condition_video_rows :]
    audio_rows = audio_rows[runner.scheduler.num_condition_audio_rows :]
    video_latents = unpatchify_video_tokens(
        video_rows,
        runner.scheduler.num_latent_frames,
        runner.scheduler.latent_height,
        runner.scheduler.latent_width,
        channels=int(runner.config.get("in_channels", 24)),
        patch_size=tuple(runner.config.get("patch_size", (1, 2, 2))),
    )
    audio_latents = unpack_audio_tokens(audio_rows, runner.scheduler.num_audio_latents)
    runner.set_vae_decode_tile_shape()
    origin = torch.cuda.current_stream()
    video_stream, audio_stream = torch.cuda.Stream(), torch.cuda.Stream()
    video_stream.wait_stream(origin)
    audio_stream.wait_stream(origin)
    try:
        with torch.cuda.stream(video_stream):
            video_latents.record_stream(video_stream)
            video = runner.video_vae.decode(video_latents, return_cpu=False)
        with torch.cuda.stream(audio_stream):
            audio_latents.record_stream(audio_stream)
            audio = runner.audio_vae.decode(audio_latents, return_cpu=False)
        origin.wait_stream(video_stream)
        origin.wait_stream(audio_stream)
        video.record_stream(origin)
        audio.record_stream(origin)
        return video, audio
    finally:
        # Completion is required before run_main's model-lifetime cleanup.
        video_stream.synchronize()
        audio_stream.synchronize()


def build(args):
    import torch

    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config

    if not torch.cuda.is_available():
        raise ValueError("A CUDA GPU is required; this benchmark never substitutes synthetic timings")
    torch.cuda.set_device(args.device)
    causal = args.task == "refa2v"
    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    parallel = cfg.get("parallel") or {}
    if any(int(parallel.get(key, 1)) != 1 for key in ("tensor_p_size", "seq_p_size", "cfg_p_size", "pp_size")):
        raise ValueError("Config requests multiple ranks. Use a single-device config for this benchmark.")
    if (cfg.get("action_prompt_travel") or {}).get("enabled", False):
        raise ValueError("Action prompt travel is not supported by this controlled benchmark")
    for key in ("dit_disk_streaming", "shared_cpu_weights", "video_vae_shared_cpu_weights", "lazy_load", "unload_modules"):
        if cfg.get(key, False):
            raise ValueError(f"{key}=true is incompatible with this benchmark")
    size = args.size or cfg.get("size") or ([256, 448] if causal else [544, 960])
    frames = args.frames if args.frames is not None else cfg.get("num_frames", 124)
    steps = args.steps if args.steps is not None else cfg.get("infer_steps", 4 if causal else 29)
    if frames < 22 or (frames - 5) % 17 or len(size) != 2 or any(v <= 0 or v % 32 for v in size) or steps < 1:
        raise ValueError("Require frames=17*n+5 (n>=1), positive steps, and canvas dimensions divisible by 32")
    cfg.update(
        model_cls="minimax_h3_causal" if causal else "minimax_h3",
        task=args.task,
        model_path=str(Path(args.model_path).expanduser().resolve()),
        num_frames=frames,
        size=size,
        infer_steps=steps,
        fps=24,
        parallel={},
        tensor_parallel=False,
        seq_parallel=False,
        cfg_parallel=False,
        warmup=False,
        h3_profile_modules=False,
        vae_cpu_offload=False,
        vae_decode_parallel=False,
        vae_encode_parallel=False,
        text_encoder_tensor_parallel=False,
    )
    if args.offload_granularity:
        cfg["offload_granularity"] = args.offload_granularity
    if args.attn_type:
        cfg["attn_type"] = args.attn_type
    if args.rms_type:
        cfg["rms_type"] = args.rms_type
    if args.rope_type:
        cfg["rope_type"] = args.rope_type
    if args.use_compile:
        cfg["use_compile"] = True
    if args.compile_backend:
        cfg["compile_backend"] = args.compile_backend
    if args.use_fused_qkv:
        cfg["use_fused_qkv"] = True
    if args.use_fused_qkv_norm_rope:
        cfg["use_fused_qkv_norm_rope"] = True
    if args.qkv_norm_rope_type:
        cfg["qkv_norm_rope_type"] = args.qkv_norm_rope_type
    if args.dit_prepost_resident:
        cfg["dit_prepost_resident"] = True
    if args.dit_release_block_offload_buffers:
        cfg["dit_release_block_offload_buffers"] = True
    cfg.setdefault("cpu_offload", True)
    cfg.setdefault("offload_granularity", "model")
    cfg.setdefault("text_encoder_cpu_offload", True)
    cfg.setdefault("attn_type", "torch_sdpa")
    cfg.setdefault("rms_type", "torch_native")
    cfg.setdefault("rope_type", "torch_real_rope")
    cfg.setdefault("feature_caching", "NoCaching")
    cfg.setdefault("enable_cfg", False)
    cfg["use_adaln_cache"] = True
    cfg.setdefault("use_compile", False)
    cfg.setdefault("video_flow_shift", 12.0)
    cfg.setdefault("audio_flow_shift", 3.0)
    cfg.setdefault("audio_sampling_rate", 32000)
    cfg.setdefault("vae_spatial_scale_factor", 16)
    if causal:
        checkpoint = args.dit_checkpoint or cfg.get("dit_original_ckpt")
        if not checkpoint or not Path(checkpoint).expanduser().exists():
            raise ValueError("refa2v requires an existing causal full checkpoint via --dit-checkpoint")
        cfg["dit_original_ckpt"] = str(Path(checkpoint).expanduser().resolve())
        cfg["h3_step_update"] = "training_euler"
        cfg.setdefault("ar_config", {"num_frame_per_chunk": 5, "sink_chunk_num": 1, "latest_chunk_num": 2, "kv_cache_max_t_rope": 7})
        for path in (args.image, args.audio):
            if not Path(path).is_file():
                raise ValueError(f"Input file does not exist: {path}")
        # Fail before expensive cache/model loading if the audio codec is absent.
        import torchaudio

        install_audio_load_fallback(torchaudio)
        try:
            torchaudio.load(args.audio)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Cannot decode driving audio. Install a TorchCodec build matching torch/FFmpeg (CPU is sufficient).") from exc
    else:
        cfg["model_variant"] = "ref2av" if args.task == "ref2av" else "fl2av"
    config = build_startup_config(cfg)
    # Use the repository's cache builder. Never overwrite an existing cache or
    # reuse another checkpoint's modulation tables, even for a proxy experiment.
    from lightx2v.models.networks.minimax_h3.adaln_cache import _build_spec, _cache_path, _validate_cache
    from tools.cache_minimax_h3_adaln.builder import build_persistent_adaln_cache

    checkpoint = Path(config["dit_original_ckpt"])
    files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
    if not files or any(not path.is_file() for path in files):
        raise ValueError(f"Missing checkpoint weights: {checkpoint}")
    identity = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    fingerprint = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    config["adaln_cache_dir"] = str(args.cache_dir.expanduser().resolve() / fingerprint)
    cfg["adaln_cache_dir"] = config["adaln_cache_dir"]
    spec, cache_path = _build_spec(config), _cache_path(config)
    if not _validate_cache(cache_path, spec):
        if cache_path.exists():
            raise ValueError(f"Invalid existing cache: {cache_path}; choose a different --cache-dir")
        print(f"BENCH preparing AdaLN cache: {cache_path} (excluded from timings)", flush=True)
        build_persistent_adaln_cache(config)
    print("BENCH configuration " + json.dumps(cfg, ensure_ascii=False), flush=True)
    if args.base_causal_proxy:
        print("BENCH WARNING: non-causal base weights in a causal execution graph; scheduling results only, no causal generation-quality claim.", flush=True)
    print("BENCH loading models (excluded from timings)", flush=True)
    runner = build_runner(config)
    request = {"task": args.task, "prompt": args.prompt, "seed": args.seed, "size": size, "num_frames": frames, "return_result_tensor": True}
    if args.task in ("refa2v", "ref2av"):
        request.update(image_path=str(Path(args.image).resolve()), audio_path=str(Path(args.audio).resolve()))
    return runner, request, cfg


def run_once(runner, request, mode, args):
    features = CAUSAL_MODES.get(mode, frozenset())
    with resident_dit(runner, "resident" in features) as policy:
        stats, result = run_request_timed(runner, request, mode, features, args)
    stats.update(policy)
    stats["features"] = sorted(features)
    return stats, result


def run_request_timed(runner, request, mode, features, args):
    from contextlib import ExitStack

    import torch

    from lightx2v.utils.profiler import no_sync_profiling

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    info = runner.prepare_request(dict(request))
    trace = Trace(torch)
    original_segment = runner.run_segment

    def segment(self, segment_idx=0):
        trace.segment_start = time.perf_counter()
        try:
            if args.task == "refa2v" and features & {"video", "audio", "early_audio"}:
                return causal_segment(self, trace, features, args.queue_depth)
            return original_segment(segment_idx)
        finally:
            trace.segment_end = time.perf_counter()

    with ExitStack() as stack:
        stack.enter_context(no_sync_profiling())
        weight_policy = {}
        if "fast_offload" in features:
            weight_policy = stack.enter_context(immutable_dit_offload(runner))
        hidden_policy = {}
        if "hidden_reuse" in features:
            hidden_policy = stack.enter_context(reuse_causal_hidden_buffer(runner))
        condition_policy = {}
        if "condition_cache_reuse" in features:
            condition_policy = stack.enter_context(reuse_causal_condition_cache(runner))
        forward_policy = {}
        if "video_head" in features:
            forward_policy = stack.enter_context(causal_video_only_head(runner))
        if "schedule" in features:
            stack.enter_context(cache_causal_schedule(runner.scheduler))
        stack.enter_context(replace_attribute(runner, "run_segment", MethodType(segment, runner)))
        if args.task == "refa2v":
            stack.enter_context(replace_attribute(runner, "stream_callback", trace.callback))
        if mode == "vae_overlap":
            stack.enter_context(replace_attribute(runner, "run_vae_decoder", MethodType(parallel_vae_decode, runner)))
        for obj, method, label in (
            (runner.model, "infer", "dit"),
            (runner.audio_vae, "encode", "audio_encode"),
            (runner.audio_vae, "decode", "audio_decode"),
            (runner.video_vae.decoder, "forward", "video_decode"),
            (runner, "run_input_encoder", "input_encode"),
            (runner, "init_run", "prepare_dit"),
            (runner, "_offload_transformer", "offload_dit"),
            (runner, "run_vae_decoder", "final_decode"),
            (runner.scheduler, "prepare", "scheduler_prepare"),
            (runner.scheduler, "step_pre", "step_pre"),
        ):
            stack.enter_context(replace_attribute(obj, method, trace.wrap(label, getattr(obj, method))))
        if args.task == "refa2v":
            for name in ("prepare_chunk", "prepare_condition"):
                stack.enter_context(replace_attribute(runner.scheduler, name, trace.wrap(name, getattr(runner.scheduler, name))))
        result = runner.run_request(info)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - trace.start) * 1000
    stats = trace.finish(wall_ms)
    stats["weight_policy"] = weight_policy
    stats["hidden_policy"] = hidden_policy
    stats["condition_policy"] = condition_policy
    stats["forward_policy"] = forward_policy
    stats["mode"] = mode
    stats["video_shape"] = list(result["video"].shape)
    stats["audio_shape"] = list(result["audio"].shape)
    if args.task == "refa2v":
        cursor = 0
        for delivery in stats["deliveries"]:
            if delivery["start_frame"] != cursor:
                raise AssertionError("RGB callback frames are missing, duplicated or out of order")
            cursor += delivery["frames"]
        if cursor != request["num_frames"] or not stats["deliveries"][-1]["is_final"]:
            raise AssertionError("Incomplete RGB stream")
    return stats, result


def compare(reference, candidate, atol, rtol):
    import torch

    comparison = {}
    for name in ("video", "audio"):
        a, b = reference[name], candidate[name]
        if a.shape != b.shape:
            comparison[name] = {"passed": False, "reference_shape": list(a.shape), "candidate_shape": list(b.shape)}
            continue
        # Bounded CPU scratch space, outside every measured interval.
        a, b = a.reshape(-1), b.reshape(-1)
        max_error, error_sum, count, passed = 0.0, 0.0, 0, True
        for offset in range(0, a.numel(), 1_000_000):
            x, y = a[offset : offset + 1_000_000].float(), b[offset : offset + 1_000_000].float()
            error = (x - y).abs()
            finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            passed = passed and finite and bool((error <= atol + rtol * x.abs()).all())
            max_error = max(max_error, float(error.max())) if finite else float("inf")
            error_sum += float(error.double().sum())
            count += x.numel()
        comparison[name] = {
            "passed": passed,
            "max_abs": max_error if math.isfinite(max_error) else None,
            "mean_abs": error_sum / max(count, 1) if math.isfinite(error_sum) else None,
        }
    comparison["passed"] = all(value["passed"] for value in comparison.values())
    return comparison


def summarize(rows):
    summary = {}
    for mode in dict.fromkeys(row["mode"] for row in rows):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {}
        for key in ("pipeline_ms", "forward_ms", "segment_ms", "first_rgb_ms", "first_rgb_from_segment_ms", "peak_allocated_gib", "peak_reserved_gib", "resident_setup_ms", "resident_cleanup_ms"):
            values = [row[key] for row in selected if row[key] is not None]
            summary[mode][key] = statistics.median(values) if values else None
        gaps = [gap for row in selected for gap in row["chunk_gaps_ms"]]
        summary[mode]["chunk_gap_p50_ms"] = percentile(gaps, 0.5)
        summary[mode]["chunk_gap_p95_ms"] = percentile(gaps, 0.95)
        summary[mode]["correct"] = all(row["comparison"]["passed"] for row in selected)
        summary[mode]["pipeline_min_ms"] = min(row["pipeline_ms"] for row in selected)
        summary[mode]["pipeline_max_ms"] = max(row["pipeline_ms"] for row in selected)
        stage_names = set().union(*(row["stage_event_ms"] for row in selected))
        summary[mode]["stage_event_ms"] = {name: statistics.median(row["stage_event_ms"].get(name, 0.0) for row in selected) for name in sorted(stage_names)}
        summary[mode]["host_stage_ms"] = {name: statistics.median(row["host_stage_ms"].get(name, 0.0) for row in selected) for name in sorted(stage_names)}
        summary[mode]["features"] = selected[0]["features"]
    baseline = summary["baseline"]
    for value in summary.values():
        value["pipeline_speedup"] = baseline["pipeline_ms"] / value["pipeline_ms"]
        value["pipeline_reduction_percent"] = 100 * (1 - value["pipeline_ms"] / baseline["pipeline_ms"])
        value["forward_speedup"] = baseline["forward_ms"] / value["forward_ms"] if value["forward_ms"] else None
        value["first_rgb_speedup"] = baseline["first_rgb_ms"] / value["first_rgb_ms"] if value["first_rgb_ms"] else None
    return summary


def main():
    args = arguments()
    import torch
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    runner, request, config = build(args)
    rows, reference = [], None
    skipped, warmed = {}, defaultdict(int)
    # Warm every variant. If warmup=0, an explicit untimed baseline still
    # supplies the correctness oracle; report this so it is not a cold test.
    for mode in args.modes:
        if mode in skipped:
            continue
        for index in range(max(args.warmup, int(mode == "baseline"))):
            print(f"BENCH warmup {mode} {index + 1}", flush=True)
            try:
                _, result = run_once(runner, request, mode, args)
            except torch.cuda.OutOfMemoryError as error:
                if mode == "baseline":
                    raise
                reason = str(error)
                skipped[mode] = {"reason": "CUDA OOM during warmup", "detail": reason}
                if "resident" in CAUSAL_MODES.get(mode, ()):
                    for other in args.modes:
                        if "resident" in CAUSAL_MODES.get(other, ()) and other != mode:
                            skipped[other] = {"reason": f"Skipped after {mode} OOM: same DiT/text-encoder residency requirement"}
                runner.clear_warmup_state()
                torch.cuda.empty_cache()
                print(f"BENCH SKIPPED {mode}: CUDA OOM; no latency result and no shape/dtype fallback", flush=True)
                break
            warmed[mode] += 1
            if mode == "baseline" and reference is None:
                reference = result
            else:
                check = compare(reference, result, args.atol, args.rtol)
                if not check["passed"]:
                    if mode == "baseline":
                        raise RuntimeError(f"Baseline is not self-consistent across warmup requests: {check}")
                    skipped[mode] = {"reason": "Warmup output mismatch; latency result discarded", "comparison": check}
                    runner.clear_warmup_state()
                    torch.cuda.empty_cache()
                    print(f"BENCH SKIPPED {mode}: output mismatch; no latency result", flush=True)
                    del result
                    break
            del result
    active_modes = [mode for mode in args.modes if mode not in skipped]
    for repetition in range(args.repeats):
        # Alternate the order to reduce thermal/cache/order bias.
        order = active_modes if repetition % 2 == 0 else list(reversed(active_modes))
        for mode in order:
            print(f"BENCH measured {repetition + 1}/{args.repeats} {mode}", flush=True)
            stats, result = run_once(runner, request, mode, args)
            stats["comparison"] = compare(reference, result, args.atol, args.rtol)
            stats["repeat"] = repetition
            rows.append(stats)
            del result
            print("BENCH result " + json.dumps({key: stats[key] for key in ("mode", "pipeline_ms", "first_rgb_ms", "chunk_gap_p50_ms", "peak_allocated_gib", "comparison")}), flush=True)
    report = {
        "schema_version": 2,
        "status": "completed_with_skips" if skipped else "complete",
        "requested_modes": args.modes,
        "skipped_modes": skipped,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task": args.task,
        "configuration": config,
        "request": request,
        "causal_weight_kind": ("noncausal_base_proxy" if args.base_causal_proxy else "user_supplied_causal_checkpoint") if args.task == "refa2v" else None,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup_per_mode": args.warmup,
        "actual_warmup_requests": {mode: warmed[mode] for mode in args.modes},
        "repeats": args.repeats,
        "notes": [
            "Real single-device executions; no sleep-based or analytical speedup estimates.",
            "All modes use resident VAEs. Baseline preserves the original runner forward order.",
            "Pipeline includes input encoders, DiT, CPU RGB delivery and final audio decoding; excludes loading/warmup/MP4/checks.",
            "Forward spans run_segment through CPU AV return, excluding input encoding/init_run; causal condition-cache prefill is included.",
            "Segment is the host wall time inside run_segment, including condition prefill and CPU RGB delivery, but excluding model offload and final audio decode.",
            "First RGB is a CPU-ready callback timestamp, not a client/network/video-player timestamp; unavailable for non-causal modes.",
            "CUDA event stage durations include contention/waits and must not be summed to infer overlap savings.",
            "Causal variants also defer small audio latent CPU copies until the end to avoid synchronizing DiT.",
            "Audio prefetch uploads the known waveform once but encodes each original truncated prefix independently.",
            "Early audio encodes every original truncated prefix before condition prefill on a side stream; assumes the driving file is already available.",
            "Scheduler metadata/noise caches are rebuilt within every timed request; no prompt/KV/output reuse across requests.",
            "Fast offload reuses existing immutable pinned CPU DiT weights; GPU copies are discarded without D2H. H2D transfers and input encoding remain inside request timing.",
            "Hidden reuse changes only causal H3's temporary packed input buffer: one full clone is avoided per denoising step, while the video rows are updated in place. Correctness is checked against baseline.",
            "Video-head mode keeps the exact final norm/modulation and video projection order, then omits the unused causal audio projection. Correctness is checked against baseline.",
            "Condition-cache reuse is an explicit probe only: it copies the step-0 causal condition cache and ring metadata, then the warmup correctness gate rejects it if per-step condition K/V differ.",
            "Forward-async mode moves causal latent unpatchifying to the existing VAE stream after a CUDA event; the next chunk's DiT submission can overlap that work. The source latent storage is retained with record_stream until consumption.",
            "--use-compile, --use-fused-qkv and --use-fused-qkv-norm-rope change the actual DiT backend at model construction. Compile/warmup work is excluded, and warmup output mismatches are reported as skipped unsafe variants.",
            "Resident modes keep DiT on GPU throughout text encoding/decoding and require extra headroom. Reported setup/cleanup are excluded from warm-service latency, not free cold-start work.",
            "No action prompt travel, live microphone availability, distributed VAE, or NCCL overlap is simulated.",
        ],
        "summary": summarize(rows),
        "runs": rows,
    }
    print("BENCH SUMMARY\n" + json.dumps(report["summary"], indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"BENCH report: {args.output.resolve()}", flush=True)
    return 0 if all(row["comparison"]["passed"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
