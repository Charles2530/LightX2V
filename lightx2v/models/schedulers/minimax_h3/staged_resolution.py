"""Opt-in H3 quality ablations; the ordinary scheduler path is untouched."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.packing import build_packed_sequence, patchify_video_latents, unpatchify_video_tokens
from lightx2v.models.schedulers.minimax_h3.changing_resolution import MiniMaxH3ChangingResolutionScheduler, resize_spatial
from lightx2v.models.schedulers.minimax_h3.resampling import flow_predict_clean, flow_renoise
from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler, _layout_to_device, _make_schedule


class MiniMaxH3StagedResolutionScheduler(MiniMaxH3ChangingResolutionScheduler):
    """Optional intermediate grid, full-canvas conditions and audio diagnostics.

    cr_audio_replay is a caller-supplied CPU tensor of baseline audio states.
    Replay requires a separate teacher rollout and is never deployment timing.
    """

    def __init__(self, config):
        # The experiment caller builds exact AdaLN entries for off-grid times.
        # Production runners must supply those entries before enabling a bump.
        cache_config = {**config, "use_adaln_cache": False} if config.get("cr_sigma_bump", 0) else config
        super().__init__(cache_config)
        self.config = config
        self.intermediate_rate = config.get("cr_intermediate_rate")
        if self.intermediate_rate is not None and not self.rate < float(self.intermediate_rate) < 1:
            raise ValueError("cr_intermediate_rate must exceed the initial rate and stay below 1")
        if self.intermediate_rate is not None and self.switch_step != 2:
            raise ValueError("Intermediate grid currently requires one step per low grid, then two high steps")
        self.keep_conditions = bool(config.get("cr_keep_full_condition", False))
        self.blur = bool(config.get("cr_upsample_blur", False))
        if self.blur:
            self.upsample_mode = "bilinear"
        self.audio_mode = config.get("cr_audio_mode", "baseline")
        if self.audio_mode not in ("baseline", "replay", "renoise", "freeze"):
            raise ValueError("Unsupported cr_audio_mode")
        if (self.sigma_bump or self.audio_mode == "freeze") and config.get("use_adaln_cache", False) and not config.get("cr_extended_adaln_cache", False):
            raise ValueError("Changed AV schedules require exact cr_extended_adaln_cache entries")
        self.original_audio_sigmas = self.audio_sigmas.clone()
        self.original_audio_timesteps = self.audio_timesteps.clone()
        self.cr_audio_replay = None
        self.grid_history = []

    def prepare(self, *args, **kwargs):
        self.audio_sigmas.copy_(self.original_audio_sigmas)
        self.audio_timesteps.copy_(self.original_audio_timesteps)
        super().prepare(*args, **kwargs)
        self.tags = self.full_state["layout_cpu"].token_tags[self.full_state["layout_cpu"].text_indices].clone()
        self.anchors = kwargs.get("keyframe_anchors", ())
        self.audio_generator = torch.Generator(device="cpu").manual_seed(int(self.generator.initial_seed()) + 999983)
        self.grid_history = [(self.latent_height, self.latent_width)]
        if self.keep_conditions:
            self._set_grid(self.video_latents[self.num_condition_video_rows :], self.latent_height, self.latent_width)
        if self.audio_mode == "freeze":
            sigmas, _ = _make_schedule(self.infer_steps - self.switch_step, self.audio_shift, self.audio_sigmas.device)
            self.audio_sigmas[: self.switch_step] = 1
            self.audio_sigmas[self.switch_step :] = sigmas
            self.audio_timesteps.copy_(1 - self.audio_sigmas[:-1])
        if self.audio_mode == "replay":
            if self.cr_audio_replay is None or self.cr_audio_replay.shape[0] != self.infer_steps + 1:
                raise ValueError("Replay needs the original four-step baseline audio trajectory, including initial/final states")
            if tuple(self.cr_audio_replay.shape[1:]) != tuple(self.audio_latents.shape):
                raise ValueError("Replay audio geometry differs from this request")

    def _set_grid(self, target_rows, height, width):
        patch = tuple(self.config.get("patch_size", (1, 2, 2)))
        full = self.full_state
        conditions = self.full_condition_rows
        if self.anchors and not self.keep_conditions:
            image = unpatchify_video_tokens(conditions, len(self.anchors), full["latent_height"], full["latent_width"], int(self.config.get("in_channels", 24)), patch)
            conditions = patchify_video_latents(resize_spatial(image, height, width, "area"), patch)
        layout = build_packed_sequence(self.tags, self.num_latent_frames, height, width, self.num_audio_latents, patch, self.anchors)
        full_layout = full["layout_cpu"]
        # The area-normalized grid is mapped to the same full-canvas footprint.
        fh, fw = full["latent_height"], full["latent_width"]
        for axis, scale in [(1, (fh / fw) ** 0.5 / (height / width) ** 0.5), (2, (fw / fh) ** 0.5 / (width / height) ** 0.5)]:
            layout.position_ids[layout.video_indices, axis] = (layout.position_ids[layout.video_indices, axis] - 16) * scale + 16
        layout.position_ids[layout.audio_indices] = full_layout.position_ids[full_layout.audio_indices]
        layout.position_ids[layout.text_indices] = full_layout.position_ids[full_layout.text_indices]
        if self.keep_conditions and self.anchors:
            ntext, ncond, naudio = self.tags.numel(), conditions.shape[0], self.audio_latents.shape[0]
            old_cond = layout.num_condition_video_rows
            positions = torch.cat(
                (
                    layout.position_ids[:ntext],
                    full_layout.position_ids[full_layout.video_indices[:ncond]],
                    layout.position_ids[layout.audio_indices],
                    layout.position_ids[layout.video_indices[old_cond:]],
                )
            )
            tags = torch.cat((self.tags, torch.zeros(ncond, dtype=torch.long), torch.full((naudio,), 2, dtype=torch.long), torch.zeros(target_rows.shape[0], dtype=torch.long)))
            video_indices = torch.cat((torch.arange(ntext, ntext + ncond), torch.arange(ntext + ncond + naudio, len(tags))))
            layout = replace(
                layout,
                sequence_length=len(tags),
                position_ids=positions,
                token_tags=tags,
                video_indices=video_indices,
                audio_indices=torch.arange(ntext + ncond, ntext + ncond + naudio),
                num_condition_video_rows=ncond,
            )
        self.video_latents = torch.cat((conditions, target_rows))
        self.layout_cpu = layout
        self.layout = _layout_to_device(layout, target_rows.device)
        self.latent_height, self.latent_width = height, width
        self.num_condition_video_rows = conditions.shape[0]

    def step_post(self):
        intermediate = self.intermediate_rate is not None and self.step_index == 0 and not self.lifted
        switching = not self.lifted and self.step_index + 1 == self.switch_step
        audio_clean = None
        if switching and self.audio_mode == "renoise":
            audio_clean = flow_predict_clean(self.audio_latents, self.audio_noise_pred, self.audio_sigmas[self.step_index])
        if switching and self.blur:
            # Modify the interpolated clean endpoint, never interpolate noise.
            start = self.num_condition_video_rows
            clean = flow_predict_clean(self.video_latents[start:], self.video_noise_pred[start:], self.video_sigmas[self.step_index])
            clean = unpatchify_video_tokens(clean, self.num_latent_frames, self.latent_height, self.latent_width, int(self.config.get("in_channels", 24)))
            h, w = self.full_state["latent_height"], self.full_state["latent_width"]
            image = resize_spatial(clean, h, w, "bilinear")
            b, c, f, _, _ = image.shape
            frames = image.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            kernel = image.new_tensor([1, 2, 1]) / 4
            weight = (kernel[:, None] * kernel[None, :])[None, None].expand(c, 1, 3, 3)
            blurred = F.conv2d(F.pad(frames, (1, 1, 1, 1), mode="replicate"), weight, groups=c).reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)
        if intermediate:
            clean = flow_predict_clean(self.video_latents[self.num_condition_video_rows :], self.video_noise_pred[self.num_condition_video_rows :], self.video_sigmas[0])
            clean = unpatchify_video_tokens(clean, self.num_latent_frames, self.latent_height, self.latent_width, int(self.config.get("in_channels", 24)))
            MiniMaxH3Scheduler.step_post(self)
            h = round(self.full_state["latent_height"] * self.intermediate_rate / 2) * 2
            w = round(self.full_state["latent_width"] * self.intermediate_rate / 2) * 2
            clean = patchify_video_latents(resize_spatial(clean, h, w, self.upsample_mode))
            noise = torch.randn(clean.shape, generator=self.lift_generator, dtype=torch.float32).to(clean.device)
            self._set_grid(flow_renoise(clean, noise, self.video_sigmas[1]), h, w)
            self.grid_history.append((h, w))
            self.video_noise_pred = self.audio_noise_pred = None
        else:
            # Replay/freeze must also operate on all earlier low steps.
            super().step_post()
        if switching and self.sigma_bump:
            index = self.step_index + 1
            ratio = self.video_sigmas[index] / self.original_video_sigmas[index]
            self.video_sigmas[index + 1 :] = self.original_video_sigmas[index + 1 :] * ratio
            self.video_timesteps.copy_(1 - self.video_sigmas[:-1])
        if switching and self.blur:
            # super used the same fresh-noise draw: recover that noise exactly
            # from the sampled state, and replace only the clean component.
            sigma = self.video_sigmas[self.step_index + 1]
            unblurred = patchify_video_latents(image)
            target = self.video_latents[self.num_condition_video_rows :]
            noise = (target - (1 - sigma) * unblurred) / sigma
            self.video_latents[self.num_condition_video_rows :] = flow_renoise(patchify_video_latents(blurred), noise, sigma)
        if self.audio_mode == "replay" and self.step_index < self.switch_step:
            self.audio_latents.copy_(self.cr_audio_replay[self.step_index + 1].to(self.audio_latents.device))
        if audio_clean is not None:
            noise = torch.randn(audio_clean.shape, generator=self.audio_generator, dtype=torch.float32).to(audio_clean.device)
            self.audio_latents.copy_(flow_renoise(audio_clean, noise, self.audio_sigmas[self.step_index + 1]))

    def clear(self):
        super().clear()
        self.audio_sigmas.copy_(self.original_audio_sigmas)
        self.audio_timesteps.copy_(self.original_audio_timesteps)
        self.cr_audio_replay = None
