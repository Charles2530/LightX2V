import argparse
import os

import torch
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import load_config
from lightx2v_train.trainers.fastwam_action_dmd.checkpoint import load_role_state_dict
from lightx2v_train.trainers.fastwam_action_dmd.config import FastWAMActionDmdConfig
from lightx2v_train.trainers.fastwam_action_dmd.roles import configure_action_role


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


def main():
    args = parse_args()
    step = infer_step_from_checkpoint_path(args.checkpoint)
    config = load_config(args.config)
    parsed = FastWAMActionDmdConfig.from_mapping(config)
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
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    module.save_checkpoint(args.output, step=step)


if __name__ == "__main__":
    main()
