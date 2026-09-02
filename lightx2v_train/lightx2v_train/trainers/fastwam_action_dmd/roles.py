import copy
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model

DEFAULT_LORA_TARGETS = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
DEFAULT_MODULES_TO_SAVE = ()


def _cast_trainable_parameters_to_fp32(expert):
    """Keep LoRA adapters and modules_to_save on FP32 optimizer state."""
    for parameter in expert.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    return expert


def _configure_role(expert, config):
    expert.requires_grad_(False)
    if config.train_type == "full":
        expert.requires_grad_(True)
        expert.train()
        return expert

    lora = config.lora
    peft_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora.get("alpha", lora["rank"])),
        lora_dropout=float(lora.get("dropout", 0.0)),
        init_lora_weights="gaussian",
        target_modules=list(lora.get("target_modules", DEFAULT_LORA_TARGETS)),
        modules_to_save=list(lora.get("modules_to_save", DEFAULT_MODULES_TO_SAVE)),
    )
    expert = get_peft_model(expert, peft_config)
    expert.train()
    return _cast_trainable_parameters_to_fp32(expert)


def configure_action_role(expert, config):
    return _configure_role(expert, config)


def configure_video_role(expert, config):
    """Configure the shared video expert for the optional video-DMD path."""
    return _configure_role(expert, config)


def attach_video_role(module, config):
    """Wrap the video expert and keep both FastWAM references in sync."""
    video_expert = configure_video_role(module.video_expert, config)
    module.video_expert = video_expert
    module.mot.mixtures["video"] = video_expert
    return video_expert


@dataclass
class ActionDmdRoles:
    student: object
    fake: object
    teacher: object

    @classmethod
    def build(cls, action_expert, parsed_config):
        teacher = copy.deepcopy(action_expert).eval()
        teacher.requires_grad_(False)
        fake = copy.deepcopy(action_expert)
        student = configure_action_role(action_expert, parsed_config.student)
        fake = configure_action_role(fake, parsed_config.fake)
        return cls(student=student, fake=fake, teacher=teacher)

    @staticmethod
    def trainable_parameters(expert):
        return [parameter for parameter in expert.parameters() if parameter.requires_grad]
