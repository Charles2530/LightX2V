from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActionDistillCondition:
    context: torch.Tensor
    context_mask: torch.Tensor
    video_kv_cache: list[dict[str, torch.Tensor] | None]
    attention_mask: torch.Tensor | list[torch.Tensor]
    video_seq_len: int

    def detach(self):
        """Return a condition detached from the video prefill autograd graph."""
        return ActionDistillCondition(
            context=self.context.detach(),
            context_mask=self.context_mask.detach(),
            video_kv_cache=[
                None if layer is None else {key: value.detach() for key, value in layer.items()}
                for layer in self.video_kv_cache
            ],
            attention_mask=(
                [mask.detach() for mask in self.attention_mask]
                if isinstance(self.attention_mask, list) else self.attention_mask.detach()
            ),
            video_seq_len=self.video_seq_len,
        )


class CachedActionDenoiser(nn.Module):
    """Run one ActionDiT expert against a shared, frozen video cache."""

    def __init__(self, expert: nn.Module, mot: nn.Module):
        super().__init__()
        self.expert = expert
        object.__setattr__(self, "_mot", mot)

    def action_module(self):
        if hasattr(self.expert, "get_base_model"):
            return self.expert.get_base_model()
        return self.expert

    def forward(self, action, timestep, condition: ActionDistillCondition):
        expert = self.action_module()
        # SparseMoT resolves the action expert from its mixtures mapping
        # instead of accepting an action_expert override.
        self._mot.mixtures["action"] = expert
        action_pre = expert.pre_dit(
            action_tokens=action,
            timestep=timestep,
            context=condition.context,
            context_mask=condition.context_mask,
        )
        tokens = self._mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=condition.video_kv_cache,
            attention_mask=condition.attention_mask,
            video_seq_len=condition.video_seq_len,
        )
        return expert.post_dit(tokens, action_pre)


def build_action_distill_condition(
    model, inputs, *, requires_grad=False, video_conditioning="observation_only", generator=None
):
    """Build the legacy observation cache or FasterWAM's one-pass future cache.

    The default remains a no-grad cache for the original action-only DMD path.
    ``torch.enable_grad`` is used explicitly for the opt-in path because callers
    may otherwise be inside a surrounding ``torch.no_grad`` context.
    """
    if video_conditioning not in {"observation_only", "one_pass_future_cache"}:
        raise ValueError(f"Unknown video_conditioning: {video_conditioning!r}")
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
    with grad_context:
        first_frame_latents = inputs["first_frame_latents"]
        batch_size = first_frame_latents.shape[0]
        future_cache = video_conditioning == "one_pass_future_cache"
        video_latents = first_frame_latents
        timestep = torch.zeros((batch_size,), device=first_frame_latents.device, dtype=first_frame_latents.dtype)
        if future_cache:
            # Use only the SHAPE of the encoded clip; never condition on GT future frames.
            shape = inputs["input_latents"].shape
            if shape[2] <= 1:
                raise ValueError("one_pass_future_cache requires a multi-frame video latent shape.")
            noise_device = first_frame_latents.device if generator is None else generator.device
            video_latents = torch.randn(shape, generator=generator, device=noise_device, dtype=torch.float32).to(first_frame_latents)
            video_latents[:, :, :1] = first_frame_latents
            timesteps, _ = model.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=1, device=first_frame_latents.device, dtype=first_frame_latents.dtype
            )
            timestep = timesteps[0].expand(batch_size)
        video_pre = model.video_expert.pre_dit(
            x=video_latents,
            timestep=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            fuse_vae_embedding_in_latents=(
                bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)) if future_cache else True
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        if future_cache:
            video_mask = model._build_video_attention_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
        else:
            video_mask = model.video_expert.build_video_to_video_mask(
                video_seq_len, int(video_pre["meta"]["tokens_per_frame"]), video_pre["tokens"].device,
            )
        video_kv_cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_mask,
        )
        build_masks = model._build_mot_attention_masks if future_cache else model._build_mot_attention_mask
        attention_mask = build_masks(
            video_seq_len=video_seq_len,
            action_seq_len=int(inputs["action"].shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=first_frame_latents.device,
        )
        return ActionDistillCondition(
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )


def sample_action_one_step(denoiser, noise, condition, num_train_timesteps):
    timestep = torch.full(
        (noise.shape[0],),
        float(num_train_timesteps),
        device=noise.device,
        dtype=noise.dtype,
    )
    return noise - denoiser(noise, timestep, condition)


@torch.no_grad()
def sample_action_teacher(denoiser, noise, condition, scheduler, num_inference_steps):
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=noise.device,
        dtype=noise.dtype,
    )
    action = noise
    for timestep, delta in zip(timesteps, deltas):
        batch_timestep = timestep.expand(action.shape[0])
        velocity = denoiser(action, batch_timestep, condition)
        action = scheduler.step(velocity, delta, action)
    return action
