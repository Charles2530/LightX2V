from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActionDistillCondition:
    context: torch.Tensor
    context_mask: torch.Tensor
    video_kv_cache: list[dict[str, torch.Tensor]]
    attention_mask: torch.Tensor
    video_seq_len: int

    def detach(self):
        """Return a condition detached from the video prefill autograd graph."""
        return ActionDistillCondition(
            context=self.context.detach(),
            context_mask=self.context_mask.detach(),
            video_kv_cache=[
                {key: value.detach() for key, value in layer.items()}
                for layer in self.video_kv_cache
            ],
            attention_mask=self.attention_mask.detach(),
            video_seq_len=self.video_seq_len,
        )


class CachedActionDenoiser(nn.Module):
    """Run one ActionDiT expert against a frozen, observation-only video cache."""

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
            action_expert=expert,
        )
        return expert.post_dit(tokens, action_pre)


def build_action_distill_condition(model, inputs, *, requires_grad=False):
    """Build the observation cache, optionally retaining gradients for video LoRA.

    The default remains a no-grad cache for the original action-only DMD path.
    ``torch.enable_grad`` is used explicitly for the opt-in path because callers
    may otherwise be inside a surrounding ``torch.no_grad`` context.
    """
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
    with grad_context:
        first_frame_latents = inputs["first_frame_latents"]
        batch_size = first_frame_latents.shape[0]
        timestep = torch.zeros((batch_size,), device=first_frame_latents.device, dtype=first_frame_latents.dtype)
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            fuse_vae_embedding_in_latents=True,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_mask = model.video_expert.build_video_to_video_mask(
            video_seq_len,
            int(video_pre["meta"]["tokens_per_frame"]),
            video_pre["tokens"].device,
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
        attention_mask = model._build_mot_attention_mask(
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
