"""One clean-latent spatial lift for H3's data-ward flow sampler.

Uses Wan's configuration names, but not its noise-prediction sign or schedule
mutation. Audio follows the original schedule, including at the video lift.
"""

import math
from numbers import Integral

import torch
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.packing import build_packed_sequence, patchify_video_latents, unpatchify_video_tokens
from lightx2v.models.schedulers.minimax_h3.resampling import flow_predict_clean, flow_renoise
from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler, _layout_to_device


def resize_spatial(latent, height, width, mode="bilinear"):
    """Resize each frame independently; never interpolate time."""
    batch, channels, frames, _, _ = latent.shape
    images = latent.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *latent.shape[-2:])
    kwargs = {} if mode in ("area", "nearest") else {"align_corners": False}
    images = F.interpolate(images.float(), size=(height, width), mode=mode, **kwargs)
    return images.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3ChangingResolutionScheduler(MiniMaxH3Scheduler):
    def __init__(self, config):
        rates = config.get("resolution_rate", [0.75])
        steps = config.get("changing_resolution_steps", [1])
        if len(rates) != 1 or len(steps) != 1:
            raise ValueError("H3 progressive resolution currently supports exactly one low-to-high transition")
        self.rate = float(rates[0])
        self.switch_step = steps[0]
        if not math.isfinite(self.rate) or not 0 < self.rate < 1:
            raise ValueError("H3 resolution_rate must be finite and strictly between 0 and 1")
        if isinstance(self.switch_step, bool) or not isinstance(self.switch_step, Integral) or not 1 <= self.switch_step < config["infer_steps"]:
            raise ValueError("H3 changing_resolution_steps counts completed low-resolution steps and must leave at least one full-resolution step")
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise ValueError("H3 progressive resolution cannot reuse feature caches across different token grids yet")
        if config.get("attn_type") == "veda_attn":
            raise ValueError("Veda's external packed-layout state is not updated by this scheduler")
        super().__init__(config)
        if self.scheduler_type != "euler":
            raise ValueError("H3 progressive resolution requires scheduler_type='euler'")
        self.upsample_mode = config.get("cr_upsample_mode", "bicubic")
        self.lowres_noise = config.get("cr_lowres_noise", "pooled")
        self.renoise_noise = config.get("cr_renoise_noise", "fresh")
        self.keep_audio_rope = config.get("cr_keep_audio_rope", True)
        self.sigma_bump = float(config.get("cr_sigma_bump", 0.0))
        if self.upsample_mode not in ("bicubic", "bilinear", "nearest"):
            raise ValueError("Unsupported cr_upsample_mode")
        if self.lowres_noise not in ("pooled", "fresh") or self.renoise_noise not in ("fresh", "initial"):
            raise ValueError("Unsupported progressive-resolution noise mode")
        if config.get("cr_condition_downsample", "area") != "area":
            raise ValueError("H3 progressive resolution currently supports area condition downsampling only")
        if not math.isfinite(self.sigma_bump) or self.sigma_bump < 0:
            raise ValueError("cr_sigma_bump must be finite and nonnegative")
        if self.sigma_bump and config.get("use_adaln_cache", False):
            raise ValueError("cr_sigma_bump changes timesteps and requires use_adaln_cache=false")
        self.original_video_sigmas = self.video_sigmas.clone()
        self.original_video_timesteps = self.video_timesteps.clone()
        self.full_state = None

    def prepare(self, seed, num_frames, height, width, text_token_tags, **kwargs):
        if any(kwargs.get(k) is not None for k in ("references", "source_video_latents", "source_audio_latents", "draft_steps")):
            raise ValueError("H3 progressive resolution does not yet support Ref2AV, suffix regeneration or draft prefixes")
        if kwargs.get("condition_audio_latents"):
            raise ValueError("H3 progressive resolution only supports target audio, not audio references")
        self.video_sigmas.copy_(self.original_video_sigmas)
        self.video_timesteps.copy_(self.original_video_timesteps)
        super().prepare(seed, num_frames, height, width, text_token_tags, **kwargs)
        patch = tuple(self.config.get("patch_size", (1, 2, 2)))
        low_h = max(patch[1], round(self.latent_height * self.rate / patch[1]) * patch[1])
        low_w = max(patch[2], round(self.latent_width * self.rate / patch[2]) * patch[2])
        if min(low_h, low_w) < 2 or (low_h, low_w) == (self.latent_height, self.latent_width):
            raise ValueError("H3 low-resolution grid collapses or does not reduce the target grid")
        self.full_state = {name: getattr(self, name) for name in ("layout", "layout_cpu", "latent_height", "latent_width", "num_condition_video_rows")}
        self.full_condition_rows = self.video_latents[: self.num_condition_video_rows].clone()
        self.full_noise_rows = self.video_latents[self.num_condition_video_rows :].clone()
        # Full prepare preserves the released audio RNG stream. Added video
        # draws use an independent stream, identical on every SP rank.
        self.lift_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
        device = self.video_latents.device
        channels = int(self.config.get("in_channels", 24))
        conditions = []
        if self.num_condition_video_rows:
            anchors = len(kwargs.get("keyframe_anchors", ()))
            condition = unpatchify_video_tokens(self.full_condition_rows, anchors, self.latent_height, self.latent_width, channels, patch)
            conditions.append(patchify_video_latents(resize_spatial(condition, low_h, low_w, "area"), patch))
        if self.lowres_noise == "pooled":
            full_noise = unpatchify_video_tokens(self.full_noise_rows, self.num_latent_frames, self.latent_height, self.latent_width, channels, patch)
            low_noise = resize_spatial(full_noise, low_h, low_w, "area")
            low_noise = (low_noise - low_noise.mean()) / low_noise.std().clamp_min(1e-8)
        else:
            low_noise = torch.randn((1, channels, self.num_latent_frames, low_h, low_w), generator=self.lift_generator, dtype=torch.float32).to(device)
        self.video_latents = torch.cat(conditions + [patchify_video_latents(low_noise, patch)])
        self.latent_height, self.latent_width = low_h, low_w
        self.layout_cpu = build_packed_sequence(text_token_tags.cpu(), self.num_latent_frames, low_h, low_w, self.num_audio_latents, patch, kwargs.get("keyframe_anchors", ()))
        # H3 positions are already area-normalized. Correct only the small
        # aspect-ratio change from snapping to patch multiples; do not multiply
        # positions by 1/rate. The continuous low/high spatial footprint matches.
        full_h, full_w = self.full_state["latent_height"], self.full_state["latent_width"]
        scale_h = (full_h / math.sqrt(full_h * full_w)) / (low_h / math.sqrt(low_h * low_w))
        scale_w = (full_w / math.sqrt(full_h * full_w)) / (low_w / math.sqrt(low_h * low_w))
        positions = self.layout_cpu.position_ids
        indices = self.layout_cpu.video_indices
        positions[indices, 1] = (positions[indices, 1] - 16.0) * scale_h + 16.0
        positions[indices, 2] = (positions[indices, 2] - 16.0) * scale_w + 16.0
        # Stereo audio positions use spatial-grid extrema in the base packer.
        # Keep them fixed to the target canvas across the resolution transition.
        if self.keep_audio_rope:
            positions[self.layout_cpu.audio_indices] = self.full_state["layout_cpu"].position_ids[self.full_state["layout_cpu"].audio_indices]
        positions[self.layout_cpu.text_indices] = self.full_state["layout_cpu"].position_ids[self.full_state["layout_cpu"].text_indices]
        self.layout = _layout_to_device(self.layout_cpu, device)
        self.num_condition_video_rows = self.layout.num_condition_video_rows
        self.lifted = False

    def step_post(self):
        if self.video_noise_pred is None or self.audio_noise_pred is None:
            raise RuntimeError("H3 progressive resolution requires both velocity predictions")
        switching = not self.lifted and self.step_index + 1 == self.switch_step
        if switching:
            start = self.num_condition_video_rows
            # Must be evaluated BEFORE step_post: v is evaluated at sigma_i,
            # not sigma_(i+1). The H3 sign is plus, unlike Wan's noise output.
            clean_rows = flow_predict_clean(self.video_latents[start:], self.video_noise_pred[start:], self.video_sigmas[self.step_index])
            clean = unpatchify_video_tokens(
                clean_rows, self.num_latent_frames, self.latent_height, self.latent_width, int(self.config.get("in_channels", 24)), tuple(self.config.get("patch_size", (1, 2, 2)))
            )
        super().step_post()  # Includes the ordinary audio update, with no lift.
        if switching:
            full = self.full_state
            clean = resize_spatial(clean, full["latent_height"], full["latent_width"], self.upsample_mode)
            patch = tuple(self.config.get("patch_size", (1, 2, 2)))
            noise = self.full_noise_rows
            if self.renoise_noise == "fresh":
                noise = torch.randn(self.full_noise_rows.shape, generator=self.lift_generator, dtype=torch.float32).to(clean.device)
            index = self.step_index + 1
            if self.sigma_bump:
                self.video_sigmas[index] = min(float(self.video_sigmas[index]) + self.sigma_bump, float(self.video_sigmas[index - 1]) - 1e-3)
                self.video_timesteps[index] = 1.0 - self.video_sigmas[index]
            high = flow_renoise(patchify_video_latents(clean, patch), noise, self.video_sigmas[index])
            self.video_latents = torch.cat((self.full_condition_rows, high))
            for name, value in full.items():
                setattr(self, name, value)
            self.video_noise_pred = None
            self.audio_noise_pred = None
            self.lifted = True

    def clear(self):
        super().clear()
        self.full_state = None
        self.full_condition_rows = None
        self.full_noise_rows = None
        self.lift_generator = None
        self.video_sigmas.copy_(self.original_video_sigmas)
        self.video_timesteps.copy_(self.original_video_timesteps)
