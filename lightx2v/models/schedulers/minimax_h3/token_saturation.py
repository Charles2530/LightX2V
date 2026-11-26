"""Token saturation for the fixed four-step H3 sampler.

Adapted from Speculative Forcing (arXiv 2610.22347) for H3's bidirectional,
single-shot packed sequence. The sigma schedule stays fixed at four steps.
After a verified forward, a target video token whose clean draft
``x0_hat = x + sigma * v`` (H3 velocity is data-ward) has stopped changing is
frozen to that draft. Later forwards no longer denoise it.

Verifier (training-free):

* Stability, after forwards listed in ``token_saturation_verify_steps``:
  ``||x0_hat^(k) - x0_hat^(k-1)||`` per token.
* Redundancy, optional and only after forward 0 with keyframes:
  ``min_h ||x0_hat^(1) - h||`` over every keyframe token ``h``.

Scores are divided by the median draft-token norm, smoothed with a separable
Gaussian over (latent frame, token row, token column) with reflect padding,
and then thresholded.

Unlike the paper's KV cache of accepted tokens, which would need about 75 GB
at 768x1344x243, frozen tokens leave the packed sequence. Frozen tokens on a
``token_saturation_context_stride`` spatial lattice stay as clean context
rows. They take the keyframe condition timestep, so they reuse the AdaLN cache
entries of conditioned requests. Stride 0 drops all frozen tokens.
"""

import math

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.networks.minimax_h3.packing import KEYFRAME_NOISE_AUG, MiniMaxH3PackedSequence
from lightx2v.models.schedulers.minimax_h3.resampling import flow_predict_clean, flow_renoise
from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler, _layout_to_device

TOKEN_SATURATION_STEPS = 4


def _gaussian_kernel(sigma: float, radius: int, device) -> torch.Tensor:
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth_axis(grid: torch.Tensor, dim: int, sigma: float) -> torch.Tensor:
    """Gaussian along one axis of a 3-D score grid, reflect padded.

    The radius is ``ceil(4 * sigma)`` (9 taps for sigma=1, as in the paper)
    and is clipped to ``size - 1``, the largest reflect padding allowed.
    """
    size = grid.shape[dim]
    radius = min(math.ceil(4.0 * sigma), size - 1)
    if sigma <= 0 or radius < 1:
        return grid
    moved = grid.movedim(dim, -1)
    shape = moved.shape
    rows = moved.reshape(-1, 1, size)
    rows = F.pad(rows, (radius, radius), mode="reflect")
    rows = F.conv1d(rows, _gaussian_kernel(sigma, radius, grid.device).view(1, 1, -1))
    return rows.reshape(shape).movedim(-1, dim)


def smooth_scores(grid: torch.Tensor, temporal_sigma: float, spatial_sigma: float) -> torch.Tensor:
    """Smooth a ``[frames, rows, cols]`` score grid before thresholding."""
    grid = _smooth_axis(grid, 0, temporal_sigma)
    grid = _smooth_axis(grid, 1, spatial_sigma)
    return _smooth_axis(grid, 2, spatial_sigma)


def _min_distance(queries: torch.Tensor, keys: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Per-query nearest-key L2 distance, chunked to bound memory."""
    return torch.cat([torch.cdist(part, keys).amin(dim=1) for part in queries.split(chunk)])


def select_layout_rows(full: MiniMaxH3PackedSequence, context_ids: torch.Tensor, active_ids: torch.Tensor) -> MiniMaxH3PackedSequence:
    """Keep text, keyframe and audio rows plus chosen target video rows.

    ``context_ids`` and ``active_ids`` index the target video rows (in target
    order). Original positions and tags are kept. Context rows are placed in
    the condition segment of ``video_indices`` so the base scheduler gives
    them the keyframe timestep and never updates them.
    """
    condition = full.num_condition_video_rows
    targets = full.video_indices[condition:]
    keep = torch.ones(full.sequence_length, dtype=torch.bool)
    keep[targets] = False
    keep[targets[context_ids]] = True
    keep[targets[active_ids]] = True
    remap = keep.long().cumsum(0) - 1
    rows = keep.nonzero().squeeze(1)
    video_indices = torch.cat((remap[full.video_indices[:condition]], remap[targets[context_ids]], remap[targets[active_ids]]))
    return MiniMaxH3PackedSequence(
        sequence_length=int(rows.numel()),
        position_ids=full.position_ids[rows],
        token_tags=full.token_tags[rows],
        video_indices=video_indices,
        audio_indices=remap[full.audio_indices],
        text_indices=remap[full.text_indices],
        num_condition_video_rows=condition + int(context_ids.numel()),
        num_condition_audio_rows=full.num_condition_audio_rows,
    )


class MiniMaxH3TokenSaturationScheduler(MiniMaxH3Scheduler):
    """Fixed four-step Euler sampler with per-token early commitment."""

    def __init__(self, config):
        if int(config["infer_steps"]) != TOKEN_SATURATION_STEPS:
            raise ValueError(f"H3 token saturation is defined for the fixed {TOKEN_SATURATION_STEPS}-step distilled schedule")
        if config.get("changing_resolution", False):
            raise ValueError("H3 token saturation cannot be combined with changing_resolution")
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise ValueError("H3 token saturation changes the packed sequence per step and requires feature_caching='NoCaching'")
        if config.get("seq_parallel", False):
            raise ValueError("H3 token saturation does not support sequence parallelism yet")
        if config.get("attn_type") == "veda_attn":
            raise ValueError("Veda's external packed-layout state is not updated by this scheduler")
        super().__init__(config)
        if self.scheduler_type != "euler":
            raise ValueError("H3 token saturation requires scheduler_type='euler'")
        self.verify_steps = tuple(sorted({int(step) for step in config.get("token_saturation_verify_steps", [1, 2])}))
        if not self.verify_steps or not all(1 <= step <= self.infer_steps - 2 for step in self.verify_steps):
            raise ValueError(f"token_saturation_verify_steps must lie in [1, {self.infer_steps - 2}]; the last forward commits every token")
        self.tau_stab = float(config.get("token_saturation_tau_stab", 0.2))
        tau_red = config.get("token_saturation_tau_red")
        self.tau_red = None if tau_red is None else float(tau_red)
        self.max_ratio = float(config.get("token_saturation_max_ratio", 0.6))
        self.context_stride = int(config.get("token_saturation_context_stride", 2))
        self.bucket = int(config.get("token_saturation_bucket", 1024))
        self.temporal_sigma, self.spatial_sigma = (float(value) for value in config.get("token_saturation_smooth_sigma", [1.0, 1.0]))
        thresholds = (self.tau_stab,) if self.tau_red is None else (self.tau_stab, self.tau_red)
        if not all(math.isfinite(tau) and tau >= 0 for tau in thresholds):
            raise ValueError("token saturation thresholds must be finite and nonnegative")
        if not 0 <= self.max_ratio < 1:
            raise ValueError("token_saturation_max_ratio must lie in [0, 1) so the last forward has tokens to denoise")
        if self.context_stride < 0 or self.bucket < 1:
            raise ValueError("token_saturation_context_stride must be >= 0 and token_saturation_bucket >= 1")
        if not all(math.isfinite(sigma) and sigma >= 0 for sigma in (self.temporal_sigma, self.spatial_sigma)):
            raise ValueError("token_saturation_smooth_sigma must be two finite, nonnegative values")
        self._reset_state()

    def _reset_state(self):
        for name in ("full_layout", "full_layout_cpu", "condition_rows", "target_rows", "initial_noise", "frozen", "scores", "prev_draft", "lattice", "active_ids", "context_ids", "grid_shape"):
            setattr(self, name, None)
        self.forward_rows = []

    def prepare(self, seed, num_frames, height, width, text_token_tags, **kwargs):
        if any(kwargs.get(key) is not None for key in ("references", "source_video_latents", "source_audio_latents", "draft_steps")):
            raise ValueError("H3 token saturation does not support Ref2AV, suffix regeneration or draft prefixes")
        super().prepare(seed, num_frames, height, width, text_token_tags, **kwargs)
        condition = self.num_condition_video_rows
        self.full_layout_cpu = self.layout_cpu
        self.full_layout = self.layout
        self.condition_rows = self.video_latents[:condition].clone()
        self.target_rows = self.video_latents[condition:].clone()
        self.initial_noise = self.target_rows.clone()
        patch_t, patch_h, patch_w = tuple(self.config.get("patch_size", (1, 2, 2)))
        self.grid_shape = (self.num_latent_frames // patch_t, self.latent_height // patch_h, self.latent_width // patch_w)
        num_targets = self.target_rows.shape[0]
        if math.prod(self.grid_shape) != num_targets:
            raise RuntimeError(f"H3 target rows {num_targets} do not match the token grid {self.grid_shape}")
        device = self.target_rows.device
        self.frozen = torch.zeros(num_targets, dtype=torch.bool, device=device)
        self.scores = torch.zeros(num_targets, dtype=torch.float32, device=device)
        self.prev_draft = None
        self.lattice = None
        if self.context_stride:
            frames, rows, cols = self.grid_shape
            on_rows = torch.arange(rows, device=device) % self.context_stride == 0
            on_cols = torch.arange(cols, device=device) % self.context_stride == 0
            self.lattice = (on_rows[:, None] & on_cols[None, :]).reshape(1, -1).expand(frames, -1).reshape(-1)
        self.active_ids = torch.arange(num_targets, device=device)
        self.context_ids = self.active_ids[:0]
        self.forward_rows = []

    def _pack_rows(self):
        """Rebuild the packed sequence from the frozen mask before a forward."""
        if not bool(self.frozen.any()):
            self.layout_cpu, self.layout = self.full_layout_cpu, self.full_layout
            self.active_ids = torch.arange(self.frozen.numel(), device=self.frozen.device)
            self.context_ids = self.active_ids[:0]
            self.video_latents = torch.cat((self.condition_rows, self.target_rows))
            return
        context = self.frozen & self.lattice if self.lattice is not None else torch.zeros_like(self.frozen)
        self.active_ids = (~self.frozen).nonzero().squeeze(1)
        self.context_ids = context.nonzero().squeeze(1)
        self.layout_cpu = select_layout_rows(self.full_layout_cpu, self.context_ids.cpu(), self.active_ids.cpu())
        self.layout = _layout_to_device(self.layout_cpu, self.frozen.device)
        # Frozen context rows enter like keyframes: clean draft at the keyframe noise level.
        context_rows = flow_renoise(self.target_rows[self.context_ids], self.initial_noise[self.context_ids], 1.0 - KEYFRAME_NOISE_AUG)
        self.video_latents = torch.cat((self.condition_rows, context_rows, self.target_rows[self.active_ids]))

    def step_pre(self, step_index, *, warmup=False):
        self._pack_rows()
        super().step_pre(step_index, warmup=warmup)
        self.forward_rows.append((int(self.active_ids.numel()), int(self.context_ids.numel())))

    def _new_frozen_count(self, num_candidates):
        """Cap the new commits by ``max_ratio`` and round the active rows up to ``bucket``."""
        total = self.frozen.numel()
        num_frozen = int(self.frozen.sum())
        count = min(num_candidates, max(int(self.max_ratio * total) - num_frozen, 0))
        active = total - num_frozen - count
        active = min(math.ceil(active / self.bucket) * self.bucket, total - num_frozen)
        return total - num_frozen - active

    def _verify(self, draft):
        """Score active tokens after forward ``step_index`` and freeze the saturated ones."""
        step = self.step_index
        scale = draft.norm(dim=1).median().clamp_min(1e-6)
        if step == 0:
            if self.tau_red is None or self.condition_rows.shape[0] == 0:
                return
            raw, tau, kind = _min_distance(draft, self.condition_rows.float()) / scale, self.tau_red, "redundancy"
        elif step in self.verify_steps:
            raw, tau, kind = (draft - self.prev_draft[self.active_ids]).norm(dim=1) / scale, self.tau_stab, "stability"
        else:
            return
        self.scores[self.active_ids] = raw
        smoothed = smooth_scores(self.scores.view(self.grid_shape), self.temporal_sigma, self.spatial_sigma).reshape(-1)
        candidates = ((smoothed < tau) & ~self.frozen).nonzero().squeeze(1)
        count = self._new_frozen_count(int(candidates.numel()))
        if count > 0:
            chosen = candidates[smoothed[candidates].argsort()[:count]]
            # ``draft`` is ordered like ``active_ids``; map chosen target ids to draft rows.
            position = torch.full_like(self.frozen, -1, dtype=torch.long)
            position[self.active_ids] = torch.arange(self.active_ids.numel(), device=position.device)
            self.target_rows[chosen] = draft[position[chosen]]
            self.frozen[chosen] = True
        if not self.is_warmup_step:
            quantiles = torch.quantile(raw.float(), torch.tensor([0.1, 0.5, 0.9], device=raw.device)).tolist()
            ratio = float(self.frozen.float().mean())
            logger.info(
                f"MiniMax-H3 token saturation step {step} ({kind}, tau={tau:g}): score q10/q50/q90={[round(q, 4) for q in quantiles]}, "
                f"candidates={int(candidates.numel())}, frozen +{count} -> {ratio:.3f}"
            )

    def step_post(self):
        condition = self.layout.num_condition_video_rows
        draft = flow_predict_clean(self.video_latents[condition:], self.video_noise_pred[condition:], self.video_sigmas[self.step_index])
        super().step_post()
        self.target_rows[self.active_ids] = self.video_latents[condition:]
        if self.step_index < self.infer_steps - 1:
            self._verify(draft)
            if self.prev_draft is None:
                self.prev_draft = torch.zeros_like(self.target_rows, dtype=torch.float32)
            self.prev_draft[self.active_ids] = draft
            return
        # Restore the full layout so VAE decode sees every target row in order.
        self.layout_cpu, self.layout = self.full_layout_cpu, self.full_layout
        self.video_latents = torch.cat((self.condition_rows, self.target_rows))
        if not self.is_warmup_step:
            total = self.frozen.numel()
            active = sum(rows for rows, _ in self.forward_rows)
            packed = sum(rows + context for rows, context in self.forward_rows)
            logger.info(
                f"MiniMax-H3 token saturation: target rows per forward {self.forward_rows}, "
                f"mean forwards per token {active / total:.3f}/{self.infer_steps}, packed target rows {packed / (total * self.infer_steps):.3f} of dense"
            )

    def clear(self):
        super().clear()
        self._reset_state()
