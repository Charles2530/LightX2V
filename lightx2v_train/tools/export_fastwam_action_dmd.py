import argparse
import os

import torch
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import load_config
from lightx2v_train.trainers.fastwam_action_dmd.checkpoint import load_role_state_dict
from lightx2v_train.trainers.fastwam_action_dmd.config import FastWAMActionDmdConfig
from lightx2v_train.trainers.fastwam_action_dmd.roles import (
    DEFAULT_LORA_TARGETS,
    DEFAULT_MODULES_TO_SAVE,
    attach_video_role,
    configure_action_role,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a FastWAM action DMD student into a native checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def infer_step_from_checkpoint_path(checkpoint_path):
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path))
    prefix = "checkpoint-"
    if not checkpoint_name.startswith(prefix):
        return None
    suffix = checkpoint_name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _role_architecture(role):
    """Return the PEFT architecture fields needed for a safe state restore."""
    if role is None:
        return None
    if role.train_type == "full":
        return ("full",)
    lora = role.lora or {}
    return (
        "lora",
        int(lora["rank"]),
        int(lora.get("alpha", lora["rank"])),
        float(lora.get("dropout", 0.0)),
        tuple(sorted(str(item) for item in lora.get("target_modules", DEFAULT_LORA_TARGETS))),
        tuple(sorted(str(item) for item in lora.get("modules_to_save", DEFAULT_MODULES_TO_SAVE))),
    )


def main():
    args = parse_args()
    step = infer_step_from_checkpoint_path(args.checkpoint)
    config = load_config(args.config)
    parsed = FastWAMActionDmdConfig.from_mapping(config)
    checkpoint_config_path = os.path.join(args.checkpoint, "config.yaml")
    if os.path.isfile(checkpoint_config_path):
        checkpoint_config = load_config(checkpoint_config_path)
        checkpoint_parsed = FastWAMActionDmdConfig.from_mapping(checkpoint_config)
        if checkpoint_parsed.unfreeze_video != parsed.unfreeze_video:
            raise RuntimeError(
                "Export config and checkpoint disagree about video training mode: "
                f"config={parsed.unfreeze_video}, checkpoint={checkpoint_parsed.unfreeze_video}."
            )
        for role_name in ("student", "fake"):
            if _role_architecture(getattr(checkpoint_parsed, role_name)) != _role_architecture(getattr(parsed, role_name)):
                raise RuntimeError(f"Export config and checkpoint disagree about {role_name} LoRA architecture.")
        if parsed.unfreeze_video and _role_architecture(checkpoint_parsed.video) != _role_architecture(parsed.video):
            raise RuntimeError("Export config and checkpoint disagree about video LoRA architecture.")
    model = build_model(config)
    model.load_components()
    module = model.unwrap_module()
    student = configure_action_role(module.action_expert, parsed.student)
    student_state = torch.load(
        os.path.join(args.checkpoint, "student_action.pt"),
        map_location="cpu",
        weights_only=True,
    )
    load_role_state_dict(student, parsed.student.train_type, student_state)
    if parsed.student.train_type == "lora":
        student = student.merge_and_unload(safe_merge=True)
    module.action_expert = student
    module.mot.mixtures["action"] = student
    if parsed.unfreeze_video:
        video_state_path = os.path.join(args.checkpoint, "video.pt")
        if not os.path.isfile(video_state_path):
            raise FileNotFoundError(
                "Video-unfreeze checkpoint is missing video.pt: "
                f"{video_state_path}"
            )
        video = attach_video_role(module, parsed.video)
        video_state = torch.load(video_state_path, map_location="cpu", weights_only=True)
        load_role_state_dict(video, parsed.video.train_type, video_state)
        if parsed.video.train_type == "lora":
            video = video.merge_and_unload(safe_merge=True)
        module.video_expert = video
        module.mot.mixtures["video"] = video
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    module.save_checkpoint(args.output, step=step)


if __name__ == "__main__":
    main()
