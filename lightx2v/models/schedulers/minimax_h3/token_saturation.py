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

Scores are divided by the median draft norm over all target tokens (frozen
tokens count with their committed drafts), smoothed with a separable Gaussian
over (latent frame, token row, token column) with reflect padding, and then
thresholded. Smoothing is a normalized convolution over active tokens only, so
frozen tokens and scores of the other verifier kind never enter it. Active
rows are rounded up to ``token_saturation_bucket`` and the context rows are
resized to the nearest multiple, so the packed target video rows are one too.

Unlike the paper's KV cache of accepted tokens, which would need about 75 GB
at 768x1344x243, frozen tokens leave the packed sequence. Frozen tokens on a
``token_saturation_context_stride`` spatial lattice stay as context rows,
recomputed every forward and never updated. Stride 0 drops all frozen tokens.
``token_saturation_context_timestep`` sets how they enter:

* ``"current"``: the frozen draft is renoised with its initial noise to the
  current sigma and takes the target timestep. Every target row then shares
  one noise level, as in training, and the AdaLN cache keys are unchanged.
* ``"keyframe"``: the draft enters like a keyframe, at the keyframe noise
  level and timestep. It reuses the AdaLN entries of conditioned requests.

Oracle mode (``token_saturation_oracle_dir``) freezes nothing, so the output
matches the dense sampler. It records every step's drafts and scores on one
trajectory, measures each token's distance to its final draft, and saves the
oracle step counts, per-step (marginal) thresholds, and a sequential replay
of the online policy on that dense trajectory, with the configured thresholds
and with thresholds calibrated step by step on the still-active tokens.
"""

import json
import math
from pathlib import Path

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


def select_layout_rows(full: MiniMaxH3PackedSequence, context_ids: torch.Tensor, active_ids: torch.Tensor, context_as_condition: bool) -> MiniMaxH3PackedSequence:
    """Keep text, keyframe and audio rows plus chosen target video rows.

    ``context_ids`` and ``active_ids`` index the target video rows (in target
    order). Original positions and tags are kept. With ``context_as_condition``
    the context rows join the condition segment of ``video_indices``, so the
    base scheduler gives them the keyframe timestep: ``[cond | context | active]``.
    Otherwise they follow the active rows in the target segment and take the
    target timestep: ``[cond | active | context]``. Callers discard whatever the
    base update writes to them.
    """
    condition = full.num_condition_video_rows
    targets = full.video_indices[condition:]
    keep = torch.ones(full.sequence_length, dtype=torch.bool)
    keep[targets] = False
    keep[targets[context_ids]] = True
    keep[targets[active_ids]] = True
    remap = keep.long().cumsum(0) - 1
    rows = keep.nonzero().squeeze(1)
    context, active = remap[targets[context_ids]], remap[targets[active_ids]]
    middle = (context, active) if context_as_condition else (active, context)
    video_indices = torch.cat((remap[full.video_indices[:condition]], *middle))
    return MiniMaxH3PackedSequence(
        sequence_length=int(rows.numel()),
        position_ids=full.position_ids[rows],
        token_tags=full.token_tags[rows],
        video_indices=video_indices,
        audio_indices=remap[full.audio_indices],
        text_indices=remap[full.text_indices],
        num_condition_video_rows=condition + (int(context_ids.numel()) if context_as_condition else 0),
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
        self.context_timestep = config.get("token_saturation_context_timestep", "current")
        if self.context_timestep not in ("current", "keyframe"):
            raise ValueError("token_saturation_context_timestep must be 'current' or 'keyframe'")
        oracle_dir = config.get("token_saturation_oracle_dir")
        self.oracle_dir = None if not oracle_dir else Path(oracle_dir).expanduser()
        self.oracle_eps = tuple(float(eps) for eps in config.get("token_saturation_oracle_eps", [0.02, 0.05, 0.1]))
        self.oracle_precision = float(config.get("token_saturation_oracle_precision", 0.95))
        if not self.oracle_eps or not all(math.isfinite(eps) and eps > 0 for eps in self.oracle_eps):
            raise ValueError("token_saturation_oracle_eps must be a nonempty list of positive values")
        if not 0 < self.oracle_precision <= 1:
            raise ValueError("token_saturation_oracle_precision must lie in (0, 1]")
        if self.oracle_dir is not None:
            logger.info(f"MiniMax-H3 token saturation oracle mode: nothing is frozen, records go to {self.oracle_dir}")
        self._reset_state()

    def _reset_state(self):
        for name in (
            "full_layout",
            "full_layout_cpu",
            "condition_rows",
            "target_rows",
            "initial_noise",
            "frozen",
            "prev_draft",
            "lattice",
            "active_ids",
            "context_ids",
            "grid_shape",
            "seed",
        ):
            setattr(self, name, None)
        self.active_start = 0
        self.forward_rows = []
        self.oracle_records = []

    def prepare(self, seed, num_frames, height, width, text_token_tags, **kwargs):
        if any(kwargs.get(key) is not None for key in ("references", "source_video_latents", "source_audio_latents", "draft_steps")):
            raise ValueError("H3 token saturation does not support Ref2AV, suffix regeneration or draft prefixes")
        super().prepare(seed, num_frames, height, width, text_token_tags, **kwargs)
        self.seed = int(seed)
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
        self.prev_draft = None
        self.lattice = None
        if self.context_stride:
            frames, rows, cols = self.grid_shape
            on_rows = torch.arange(rows, device=device) % self.context_stride == 0
            on_cols = torch.arange(cols, device=device) % self.context_stride == 0
            self.lattice = (on_rows[:, None] & on_cols[None, :]).reshape(1, -1).expand(frames, -1).reshape(-1)
        self.active_ids = torch.arange(num_targets, device=device)
        self.context_ids = self.active_ids[:0]
        self.active_start = condition
        self.forward_rows = []
        self.oracle_records = []

    def _pack_rows(self, step_index):
        """Rebuild the packed sequence from the frozen mask before a forward."""
        condition = self.condition_rows.shape[0]
        if not bool(self.frozen.any()):
            self.layout_cpu, self.layout = self.full_layout_cpu, self.full_layout
            self.active_ids = torch.arange(self.frozen.numel(), device=self.frozen.device)
            self.context_ids = self.active_ids[:0]
            self.active_start = condition
            self.video_latents = torch.cat((self.condition_rows, self.target_rows))
            return
        context = self.frozen & self.lattice if self.lattice is not None else torch.zeros_like(self.frozen)
        self.active_ids = (~self.frozen).nonzero().squeeze(1)
        self.context_ids = self._bucket_context(context)
        as_condition = self.context_timestep == "keyframe"
        self.layout_cpu = select_layout_rows(self.full_layout_cpu, self.context_ids.cpu(), self.active_ids.cpu(), as_condition)
        self.layout = _layout_to_device(self.layout_cpu, self.frozen.device)
        # "keyframe": clean draft at the keyframe noise level. "current": the
        # draft renoised with its own initial noise to this step's sigma.
        sigma = 1.0 - KEYFRAME_NOISE_AUG if as_condition else self.video_sigmas[step_index]
        context_rows = flow_renoise(self.target_rows[self.context_ids], self.initial_noise[self.context_ids], sigma)
        active_rows = self.target_rows[self.active_ids]
        if as_condition:
            self.active_start = condition + context_rows.shape[0]
            self.video_latents = torch.cat((self.condition_rows, context_rows, active_rows))
        else:
            self.active_start = condition
            self.video_latents = torch.cat((self.condition_rows, active_rows, context_rows))

    @staticmethod
    def _spread(ids, count):
        """``count`` evenly spaced entries of ``ids``."""
        if count <= 0:
            return ids[:0]
        return ids[torch.arange(count, device=ids.device) * ids.numel() // count]

    def _bucket_context(self, context):
        """Context ids, resized so active + context target rows are a multiple of ``bucket``.

        Once anything is frozen the active rows are already a multiple (see
        ``_capped_count``). The lattice count goes to the nearest multiple:
        down by dropping evenly spaced lattice rows, or up by adding evenly
        spaced off-lattice frozen rows when there are enough. With stride 0
        there is no context and nothing changes.
        """
        lattice_ids = context.nonzero().squeeze(1)
        count = int(lattice_ids.numel())
        remainder = count % self.bucket
        if remainder == 0:
            return lattice_ids
        extra = (self.frozen & ~context).nonzero().squeeze(1)
        grow = self.bucket - remainder
        if 2 * remainder >= self.bucket and grow <= extra.numel():
            return torch.cat((lattice_ids, self._spread(extra, grow))).sort().values
        return self._spread(lattice_ids, count - remainder)

    def step_pre(self, step_index, *, warmup=False):
        self._pack_rows(int(step_index))
        super().step_pre(step_index, warmup=warmup)
        self.forward_rows.append((int(self.active_ids.numel()), int(self.context_ids.numel())))

    def _capped_count(self, num_frozen, num_candidates):
        """New commits allowed by ``max_ratio``, with the active rows rounded up to ``bucket``."""
        total = self.frozen.numel()
        count = min(num_candidates, max(int(self.max_ratio * total) - num_frozen, 0))
        num_active = total - num_frozen
        active = min(math.ceil((num_active - count) / self.bucket) * self.bucket, num_active)
        return num_active - active

    def _scale(self, draft):
        """Median draft norm over every target token.

        Active tokens use ``draft``; frozen rows of ``target_rows`` hold their
        committed drafts. Online and oracle scores therefore share one scale
        while the two trajectories agree.
        """
        norms = self.target_rows.float().norm(dim=1)
        norms[self.active_ids] = draft.float().norm(dim=1)
        return norms.median().clamp_min(1e-6)

    def _masked_smooth(self, scores, active):
        """Gaussian-smooth scores over active tokens only (normalized convolution).

        Frozen tokens get zero weight and are never candidates, so stale or
        foreign scores cannot leak into their neighbours. With every token
        active this equals plain smoothing. Always returns a new tensor.
        """
        mask = active.float().view(self.grid_shape)
        weighted = smooth_scores(scores.view(self.grid_shape) * mask, self.temporal_sigma, self.spatial_sigma)
        weight = smooth_scores(mask, self.temporal_sigma, self.spatial_sigma)
        return (weighted / weight.clamp_min(1e-12)).reshape(-1)

    def _choose(self, smoothed, active, tau, num_frozen):
        """Active tokens to freeze: lowest smoothed scores below ``tau``, capped by ``max_ratio``."""
        candidates = (active & (smoothed < tau)).nonzero().squeeze(1)
        count = self._capped_count(num_frozen, int(candidates.numel()))
        return candidates[smoothed[candidates].argsort()[:count]], int(candidates.numel())

    def _verify(self, draft):
        """Score active tokens after forward ``step_index`` and freeze the saturated ones."""
        step = self.step_index
        if step == 0:
            if self.tau_red is None or self.condition_rows.shape[0] == 0:
                return
            kind, tau = "redundancy", self.tau_red
            raw = _min_distance(draft, self.condition_rows.float()) / self._scale(draft)
        elif step in self.verify_steps:
            kind, tau = "stability", self.tau_stab
            raw = (draft - self.prev_draft[self.active_ids]).norm(dim=1) / self._scale(draft)
        else:
            return
        active = ~self.frozen
        scores = torch.zeros_like(self.frozen, dtype=torch.float32)
        scores[self.active_ids] = raw
        chosen, num_candidates = self._choose(self._masked_smooth(scores, active), active, tau, int(self.frozen.sum()))
        if chosen.numel():
            # ``draft`` is ordered like ``active_ids``; map chosen target ids to draft rows.
            position = torch.full_like(self.frozen, -1, dtype=torch.long)
            position[self.active_ids] = torch.arange(self.active_ids.numel(), device=position.device)
            self.target_rows[chosen] = draft[position[chosen]].to(self.target_rows.dtype)
            self.frozen[chosen] = True
        if not self.is_warmup_step:
            quantiles = torch.quantile(raw.float(), torch.tensor([0.1, 0.5, 0.9], device=raw.device)).tolist()
            ratio = float(self.frozen.float().mean())
            logger.info(
                f"MiniMax-H3 token saturation step {step} ({kind}, tau={tau:g}): score q10/q50/q90={[round(q, 4) for q in quantiles]}, "
                f"candidates={num_candidates}, frozen +{int(chosen.numel())} -> {ratio:.3f}"
            )

    def _record_oracle(self, draft):
        """Dense trajectory: keep this step's draft and unnormalized verifier distances."""
        record = {"draft": draft.float()}
        if self.step_index > 0:
            record["stability"] = (draft - self.prev_draft).norm(dim=1)
        if self.step_index == 0 and self.condition_rows.shape[0]:
            record["redundancy"] = _min_distance(draft, self.condition_rows.float())
        self.oracle_records.append(record)

    def _calibrate(self, scores, good):
        """Largest ``tau`` whose accepted set ``scores < tau`` reaches ``oracle_precision``.

        The cut is made only after a complete group of equal scores, so the
        set ``scores < tau`` is exactly the prefix that was checked. The
        statistics are then recomputed on that set.
        """
        ranked, order = scores.sort()
        hits = good[order].float().cumsum(0)
        counts = torch.arange(1, ranked.numel() + 1, device=ranked.device)
        group_end = torch.ones_like(ranked, dtype=torch.bool)
        group_end[:-1] = ranked[1:] > ranked[:-1]
        valid = (group_end & (hits / counts >= self.oracle_precision)).nonzero()
        tau = 0.0
        if valid.numel():
            last = int(valid[-1])
            # float32 bound, so comparing float32 scores with it is exact.
            bound = ranked[last + 1] if last + 1 < ranked.numel() else torch.nextafter(ranked[last], ranked.new_tensor(math.inf))
            tau = float(bound)
        accepted = scores < tau
        num_accepted = int(accepted.sum())
        return {
            "tau": tau,
            "accepted_fraction": num_accepted / max(int(scores.numel()), 1),
            "precision": float(good[accepted].float().mean()) if num_accepted else None,
            "recall": float((accepted & good).sum() / good.sum().clamp_min(1)),
        }

    def _verify_plan(self, use_config):
        """(step, kind) pairs the online verifier runs, in order."""
        plan = []
        if "redundancy" in self.oracle_records[0] and (not use_config or self.tau_red is not None):
            plan.append((0, "redundancy"))
        return plan + [(step, "stability") for step in self.verify_steps]

    def _replay(self, rel, eps_list, eps=None):
        """Replay online freezing on the dense trajectory.

        Uses the online rules: full-token scale with committed drafts for
        frozen tokens, masked smoothing, ``max_ratio`` and ``bucket``. With
        ``eps`` each step's ``tau`` is calibrated on the still-active tokens;
        without it the configured thresholds are used. The dense trajectory
        stands in for the online one, so drift caused by freezing is not modelled.
        """
        records = self.oracle_records
        num_tokens = rel.shape[1]
        frozen = torch.zeros(num_tokens, dtype=torch.bool, device=rel.device)
        freeze_step = torch.full((num_tokens,), self.infer_steps - 1, dtype=torch.long, device=rel.device)
        committed_norm = torch.zeros(num_tokens, dtype=torch.float32, device=rel.device)
        steps = []
        for step, kind in self._verify_plan(use_config=eps is None):
            draft_norm = records[step]["draft"].norm(dim=1)
            scale = torch.where(frozen, committed_norm, draft_norm).median().clamp_min(1e-6)
            active = ~frozen
            smoothed = self._masked_smooth(records[step][kind] / scale, active)
            if eps is None:
                tau = self.tau_red if kind == "redundancy" else self.tau_stab
            else:
                tau = self._calibrate(smoothed[active], rel[step][active] <= eps)["tau"]
            chosen, num_candidates = self._choose(smoothed, active, tau, int(frozen.sum()))
            steps.append({"step": step, "kind": kind, "tau": tau, "candidates": num_candidates, "frozen_new": int(chosen.numel())})
            frozen[chosen] = True
            freeze_step[chosen] = step
            committed_norm[chosen] = draft_norm[chosen]
        # Error of each committed token against the dense final draft; unfrozen tokens end exact.
        error = torch.zeros(num_tokens, dtype=torch.float32, device=rel.device)
        error[frozen] = rel[freeze_step[frozen], frozen.nonzero().squeeze(1)]
        result = {
            "steps": steps,
            "frozen_fraction": round(float(frozen.float().mean()), 4),
            "mean_forwards": round(float((freeze_step + 1).float().mean()), 4),
            "frozen_ok_fraction": {f"{value:g}": (round(float((error[frozen] <= value).float().mean()), 4) if bool(frozen.any()) else None) for value in eps_list},
        }
        if bool(frozen.any()):
            quantiles = torch.tensor([0.5, 0.9, 0.99], device=rel.device)
            result["frozen_rel_q50_q90_q99"] = [round(q, 5) for q in torch.quantile(error[frozen], quantiles).tolist()]
        return result

    def _finish_oracle(self, final_draft):
        """Compare every draft with the final one, then save step counts and thresholds."""
        final = final_draft.float()
        scale = final.norm(dim=1).median().clamp_min(1e-6)
        rel = torch.stack([(record["draft"] - final).norm(dim=1) / scale for record in self.oracle_records])
        quantiles = torch.tensor([0.1, 0.5, 0.9, 0.99], device=rel.device)
        steps = self.infer_steps
        summary = {
            "seed": self.seed,
            "size": [self.height, self.width],
            "num_frames": self.num_frames,
            "grid_shape": list(self.grid_shape),
            "num_tokens": int(final.shape[0]),
            "video_sigmas": [round(float(sigma), 6) for sigma in self.video_sigmas],
            "rel_to_final_q10_q50_q90_q99": {f"step{step}": [round(q, 5) for q in torch.quantile(row, quantiles).tolist()] for step, row in enumerate(rel)},
            "precision_target": self.oracle_precision,
            "oracle": {},
            "calibration": {},
        }
        for eps in self.oracle_eps:
            ok = rel <= eps
            first = torch.where(ok.any(0), ok.float().argmax(0), torch.full_like(ok[0], steps - 1, dtype=torch.long))
            forwards = first + 1
            summary["oracle"][f"{eps:g}"] = {"mean_forwards": round(float(forwards.float().mean()), 4), "forwards_histogram": torch.bincount(forwards, minlength=steps + 1)[1:].tolist()}
        # Marginal calibration: every token active, each step on its own.
        tensors = {"grid_shape": self.grid_shape, "rel_to_final": rel.cpu()}
        all_active = torch.ones(rel.shape[1], dtype=torch.bool, device=rel.device)
        targets = [("stability", step) for step in range(1, steps - 1)]
        if "redundancy" in self.oracle_records[0]:
            targets.append(("redundancy", 0))
        for name, step in targets:
            scores = self.oracle_records[step][name] / self.oracle_records[step]["draft"].norm(dim=1).median().clamp_min(1e-6)
            smoothed = self._masked_smooth(scores, all_active)
            tensors[f"{name}_step{step}"], tensors[f"{name}_smoothed_step{step}"] = scores.cpu(), smoothed.cpu()
            summary["calibration"][f"{name}_step{step}"] = {f"{eps:g}": self._calibrate(smoothed, rel[step] <= eps) for eps in self.oracle_eps}
        # Sequential replay of the online policy: configured thresholds, then per-eps calibrated ones.
        summary["replay"] = {
            "config": self._replay(rel, self.oracle_eps),
            "calibrated": {f"{eps:g}": self._replay(rel, self.oracle_eps, eps=eps) for eps in self.oracle_eps},
        }
        self.oracle_dir.mkdir(parents=True, exist_ok=True)
        stem = self.oracle_dir / f"token_saturation_oracle_seed{self.seed}_{self.height}x{self.width}x{self.num_frames}"
        torch.save(tensors, stem.with_suffix(".pt"))
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        logger.info(f"MiniMax-H3 token saturation oracle (eps -> mean forwards): {json.dumps(summary['oracle'])}, saved to {stem}.json/.pt")

    def step_post(self):
        # Only the active rows are read back; context rows were never meant to move.
        active = slice(self.active_start, self.active_start + int(self.active_ids.numel()))
        draft = flow_predict_clean(self.video_latents[active], self.video_noise_pred[active], self.video_sigmas[self.step_index])
        super().step_post()
        self.target_rows[self.active_ids] = self.video_latents[active]
        if self.step_index < self.infer_steps - 1:
            if self.oracle_dir is None:
                self._verify(draft)
            else:
                self._record_oracle(draft)
            if self.prev_draft is None:
                self.prev_draft = torch.zeros_like(self.target_rows, dtype=torch.float32)
            self.prev_draft[self.active_ids] = draft
            return
        # Restore the full layout so VAE decode sees every target row in order.
        self.layout_cpu, self.layout = self.full_layout_cpu, self.full_layout
        self.video_latents = torch.cat((self.condition_rows, self.target_rows))
        if not self.is_warmup_step:
            if self.oracle_dir is not None:
                self._finish_oracle(draft)
            total = self.frozen.numel()
            forwards = sum(rows for rows, _ in self.forward_rows)
            packed = sum(rows + context for rows, context in self.forward_rows)
            logger.info(
                f"MiniMax-H3 token saturation: target rows per forward {self.forward_rows}, "
                f"mean forwards per token {forwards / total:.3f}/{self.infer_steps}, packed target rows {packed / (total * self.infer_steps):.3f} of dense"
            )

    def clear(self):
        super().clear()
        self._reset_state()
