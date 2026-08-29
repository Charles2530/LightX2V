import os

import torch
import yaml
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_main_process


def _atomic_torch_save(payload, path):
    temporary_path = f"{path}.tmp-rank-{get_rank():05d}-pid-{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _atomic_yaml_save(payload, path):
    temporary_path = f"{path}.tmp-rank-{get_rank():05d}-pid-{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _role_state_dict(expert, train_type):
    if train_type == "lora":
        return get_peft_model_state_dict(expert)
    return expert.state_dict()


def load_role_state_dict(expert, train_type, state_dict):
    if train_type == "lora":
        incompatible = set_peft_model_state_dict(expert, state_dict)
        if incompatible and incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected LoRA checkpoint keys: {incompatible.unexpected_keys}")
        return
    expert.load_state_dict(state_dict, strict=True)


class ActionDmdCheckpointManager:
    def __init__(self, trainer):
        self.trainer = trainer

    def resolve_resume(self):
        resume = self.trainer.config.get("resume", {})
        if resume.get("resume_ckpt_path"):
            path = str(resume["resume_ckpt_path"])
            return path, int(os.path.basename(path).split("-")[-1])
        if not resume.get("auto_resume", False):
            return None, 0
        return find_latest_checkpoint(self.trainer.output_dir)

    def save(self, iteration):
        trainer = self.trainer
        if is_main_process():
            prune_checkpoints(trainer.output_dir, trainer.save_total_limit)
        save_dir = os.path.join(trainer.output_dir, f"checkpoint-{iteration:09d}")
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()
        rng_state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        _atomic_torch_save(rng_state, os.path.join(save_dir, f"rng-rank-{get_rank():05d}.pt"))
        barrier()
        if is_main_process():
            _atomic_torch_save(
                _role_state_dict(trainer.roles.student, trainer.parsed.student.train_type),
                os.path.join(save_dir, "student_action.pt"),
            )
            _atomic_torch_save(
                _role_state_dict(trainer.roles.fake, trainer.parsed.fake.train_type),
                os.path.join(save_dir, "fake_action.pt"),
            )
            _atomic_yaml_save(
                trainer.runtime_config,
                os.path.join(save_dir, "config.yaml"),
            )
            # Write the completion marker last. Runtime checkpoint discovery uses
            # training_state.pt to distinguish complete checkpoints.
            _atomic_torch_save(
                {
                    "iteration": iteration,
                    "world_size": get_world_size(),
                    "student_train_type": trainer.parsed.student.train_type,
                    "fake_train_type": trainer.parsed.fake.train_type,
                    "student_optimizer": trainer.student_optimizer.state_dict(),
                    "fake_optimizer": trainer.fake_optimizer.state_dict(),
                    "student_scheduler": trainer.student_scheduler.state_dict(),
                    "fake_scheduler": trainer.fake_scheduler.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )
        barrier()

    def load(self, checkpoint_dir):
        trainer = self.trainer
        state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location="cpu", weights_only=False)
        if int(state["world_size"]) != get_world_size():
            raise RuntimeError(f"Checkpoint world_size={state['world_size']} does not match current world_size={get_world_size()}.")
        if state["student_train_type"] != trainer.parsed.student.train_type:
            raise RuntimeError("Checkpoint student train type does not match the current configuration.")
        if state["fake_train_type"] != trainer.parsed.fake.train_type:
            raise RuntimeError("Checkpoint fake train type does not match the current configuration.")
        rng_path = os.path.join(checkpoint_dir, f"rng-rank-{get_rank():05d}.pt")
        if not os.path.isfile(rng_path):
            raise RuntimeError(f"Checkpoint RNG state is missing for rank {get_rank()}: {rng_path}")
        load_role_state_dict(
            trainer.roles.student,
            trainer.parsed.student.train_type,
            torch.load(os.path.join(checkpoint_dir, "student_action.pt"), map_location="cpu", weights_only=True),
        )
        load_role_state_dict(
            trainer.roles.fake,
            trainer.parsed.fake.train_type,
            torch.load(os.path.join(checkpoint_dir, "fake_action.pt"), map_location="cpu", weights_only=True),
        )
        trainer.student_optimizer.load_state_dict(state["student_optimizer"])
        trainer.fake_optimizer.load_state_dict(state["fake_optimizer"])
        trainer.student_scheduler.load_state_dict(state["student_scheduler"])
        trainer.fake_scheduler.load_state_dict(state["fake_scheduler"])
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=True)
        torch.set_rng_state(rng_state["cpu"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"])
        return int(state["iteration"])
