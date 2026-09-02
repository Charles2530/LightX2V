from types import SimpleNamespace

import pytest
import torch
from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import (
    CachedActionDenoiser,
    build_action_distill_condition,
)
from lightx2v_train.trainers.fastwam_action_dmd.checkpoint import (
    ActionDmdCheckpointManager,
    _role_state_dict,
    load_role_state_dict,
)
from lightx2v_train.trainers.fastwam_action_dmd.config import FastWAMActionDmdConfig
from lightx2v_train.trainers.fastwam_action_dmd.roles import attach_video_role, configure_video_role
from torch import nn


def _base_config(**training_overrides):
    training = {
        "student": {
            "train_type": "lora",
            "lora": {"rank": 2},
            "optimizer": {},
        },
        "fake": {
            "train_type": "lora",
            "lora": {"rank": 2},
            "optimizer": {},
        },
        "action_dmd": {},
    }
    training.update(training_overrides)
    return {"training": training}


def test_video_unfreeze_is_disabled_for_legacy_configs():
    parsed = FastWAMActionDmdConfig.from_mapping(_base_config())

    assert parsed.unfreeze_video is False
    assert parsed.video is None
    assert parsed.video_anchor_weight == 0.0


def test_video_unfreeze_parses_a_video_role_and_anchor_weight():
    parsed = FastWAMActionDmdConfig.from_mapping(
        _base_config(
            unfreeze_video=True,
            video_anchor_weight=1.0e-4,
            video={
                "train_type": "lora",
                "lora": {"rank": 8, "target_modules": ["q"]},
                "optimizer": {"learning_rate": 1.0e-5},
            },
        )
    )

    assert parsed.unfreeze_video is True
    assert parsed.video.train_type == "lora"
    assert parsed.video.lora["rank"] == 8
    assert parsed.video_anchor_weight == pytest.approx(1.0e-4)


def test_video_unfreeze_requires_video_role_configuration():
    with pytest.raises(TypeError, match="training.video"):
        FastWAMActionDmdConfig.from_mapping(_base_config(unfreeze_video=True))


class _TinyVideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)

    def pre_dit(self, **kwargs):
        tokens = self.projection(kwargs["x"])
        batch_size, seq_len, _ = tokens.shape
        return {
            "tokens": tokens,
            "freqs": torch.zeros(seq_len, 1, 2),
            "t_mod": torch.zeros(batch_size, 1, 2),
            "context": kwargs["context"],
            "context_mask": kwargs["context_mask"],
            "meta": {"tokens_per_frame": seq_len},
        }

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        del video_tokens_per_frame
        return torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device)


class _TinyMot:
    def prefill_video_cache(self, **kwargs):
        tokens = kwargs["video_tokens"]
        return [{"k": tokens, "v": tokens * 2.0}]

    def forward_action_with_video_cache(self, *, action_tokens, video_kv_cache, **kwargs):
        del kwargs
        video_signal = video_kv_cache[0]["k"].mean(dim=1, keepdim=True)
        return action_tokens + video_signal


class _TinyActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)

    def pre_dit(self, *, action_tokens, **kwargs):
        del kwargs
        return {
            "tokens": self.projection(action_tokens),
            "freqs": torch.zeros(action_tokens.shape[1], 1, 2),
            "t_mod": torch.zeros(action_tokens.shape[0], 1, 2),
            "context": None,
            "context_mask": None,
        }

    def post_dit(self, tokens, pre_state):
        del pre_state
        return tokens


class _TinyFastWAM:
    def __init__(self):
        self.video_expert = _TinyVideoExpert()
        self.mot = _TinyMot()

    def _build_mot_attention_mask(self, **kwargs):
        total = kwargs["video_seq_len"] + kwargs["action_seq_len"]
        return torch.ones(total, total, dtype=torch.bool, device=kwargs["device"])


def _condition_inputs():
    return {
        "first_frame_latents": torch.ones(1, 2, 2),
        "context": torch.zeros(1, 1, 2),
        "context_mask": torch.ones(1, 1, dtype=torch.bool),
        "action": torch.zeros(1, 1, 2),
    }


def test_video_condition_is_detached_by_default_and_differentiable_when_enabled():
    model = _TinyFastWAM()

    frozen_condition = build_action_distill_condition(model, _condition_inputs())
    assert frozen_condition.video_kv_cache[0]["k"].requires_grad is False

    train_condition = build_action_distill_condition(model, _condition_inputs(), requires_grad=True)
    assert train_condition.video_kv_cache[0]["k"].requires_grad is True
    train_condition.video_kv_cache[0]["k"].sum().backward()
    assert model.video_expert.projection.weight.grad is not None


def test_student_action_loss_can_backpropagate_through_video_cache():
    model = _TinyFastWAM()
    action = _TinyActionExpert()
    condition = build_action_distill_condition(model, _condition_inputs(), requires_grad=True)
    denoiser = CachedActionDenoiser(action, model.mot)

    output = denoiser(torch.ones(1, 1, 2), torch.ones(1), condition)
    output.sum().backward()

    assert model.video_expert.projection.weight.grad is not None
    assert action.projection.weight.grad is not None


def test_condition_detach_removes_video_autograd_history():
    model = _TinyFastWAM()
    condition = build_action_distill_condition(model, _condition_inputs(), requires_grad=True)

    detached = condition.detach()

    assert detached.video_kv_cache[0]["k"].requires_grad is False
    assert detached.context.requires_grad is False


def test_video_role_configures_only_video_lora_parameters():
    expert = nn.Sequential(nn.Linear(2, 2))
    config = SimpleNamespace(
        train_type="lora",
        lora={"rank": 1, "alpha": 1, "target_modules": ["0"]},
    )

    configured = configure_video_role(expert, config)

    trainable = [name for name, parameter in configured.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_video_role_attachment_updates_the_mot_video_alias():
    expert = nn.Sequential(nn.Linear(2, 2))
    module = SimpleNamespace(video_expert=expert, mot=SimpleNamespace(mixtures={"video": expert}))
    config = SimpleNamespace(
        train_type="lora",
        lora={"rank": 1, "alpha": 1, "target_modules": ["0"]},
    )

    configured = attach_video_role(module, config)

    assert module.video_expert is configured
    assert module.mot.mixtures["video"] is configured


def test_lora_checkpoint_rejects_a_different_adapter_topology():
    source_config = SimpleNamespace(
        train_type="lora",
        lora={"rank": 1, "alpha": 1, "target_modules": ["0"]},
    )
    target_config = SimpleNamespace(
        train_type="lora",
        lora={"rank": 1, "alpha": 1, "target_modules": ["1"]},
    )
    source = configure_video_role(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), source_config)
    target = configure_video_role(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), target_config)
    state = _role_state_dict(source, "lora")

    with pytest.raises(RuntimeError, match="adapter structure"):
        load_role_state_dict(target, "lora", state)


def _checkpoint_trainer(tmp_path, *, video_enabled):
    student = nn.Linear(2, 2)
    fake = nn.Linear(2, 2)
    video = nn.Linear(2, 2) if video_enabled else None
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1.0e-3)
    fake_optimizer = torch.optim.AdamW(fake.parameters(), lr=1.0e-3)
    video_optimizer = torch.optim.AdamW(video.parameters(), lr=1.0e-4) if video is not None else None
    parsed = SimpleNamespace(
        student=SimpleNamespace(train_type="full"),
        fake=SimpleNamespace(train_type="full"),
        video=SimpleNamespace(train_type="full") if video is not None else None,
    )
    trainer = SimpleNamespace(
        config={"resume": {}},
        output_dir=str(tmp_path),
        save_total_limit=3,
        save_final=True,
        roles=SimpleNamespace(student=student, fake=fake),
        parsed=parsed,
        video_expert=video,
        video_anchor_state={"weight": torch.ones(2, 2)} if video is not None else {},
        student_optimizer=student_optimizer,
        fake_optimizer=fake_optimizer,
        video_optimizer=video_optimizer,
        student_scheduler=torch.optim.lr_scheduler.StepLR(student_optimizer, 10),
        fake_scheduler=torch.optim.lr_scheduler.StepLR(fake_optimizer, 10),
        video_scheduler=torch.optim.lr_scheduler.StepLR(video_optimizer, 10) if video_optimizer is not None else None,
        runtime_config={"training": {"unfreeze_video": video_enabled}},
    )
    trainer.checkpoints = ActionDmdCheckpointManager(trainer)
    return trainer


def test_enabled_checkpoint_saves_and_restores_video_state(tmp_path):
    trainer = _checkpoint_trainer(tmp_path, video_enabled=True)
    expected = trainer.video_expert.weight.detach().clone()
    trainer.checkpoints.save(3)
    checkpoint = tmp_path / "checkpoint-000000003"

    assert (checkpoint / "video.pt").is_file()
    assert (checkpoint / "video_anchor.pt").is_file()
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["video_enabled"] is True
    assert state["video_optimizer"] is not None

    restored = _checkpoint_trainer(tmp_path / "restored", video_enabled=True)
    restored.output_dir = str(tmp_path / "restored")
    iteration = restored.checkpoints.load(checkpoint)

    assert iteration == 3
    assert torch.equal(restored.video_expert.weight, expected)
    assert restored.video_anchor_state["weight"].shape == (2, 2)


def test_enabled_checkpoint_without_video_state_fails_loudly(tmp_path):
    trainer = _checkpoint_trainer(tmp_path, video_enabled=True)
    trainer.checkpoints.save(1)
    checkpoint = tmp_path / "checkpoint-000000001"
    (checkpoint / "video.pt").unlink()
    restored = _checkpoint_trainer(tmp_path / "restored", video_enabled=True)

    with pytest.raises(RuntimeError, match="missing video state"):
        restored.checkpoints.load(checkpoint)
