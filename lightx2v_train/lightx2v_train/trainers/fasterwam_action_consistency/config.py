from dataclasses import dataclass


@dataclass(frozen=True)
class ActionStudentConfig:
    train_type: str
    optimizer: dict
    lora: dict | None

    @classmethod
    def from_mapping(cls, mapping):
        train_type = str(mapping.get("train_type", "lora")).lower()
        if train_type not in {"lora", "full"}:
            raise ValueError(f"training.student.train_type must be 'lora' or 'full', got {train_type!r}.")
        lora = mapping.get("lora")
        if train_type == "lora" and (not isinstance(lora, dict) or int(lora.get("rank", 0)) <= 0):
            raise ValueError("training.student.lora with a positive rank is required for LoRA training.")
        optimizer = mapping.get("optimizer")
        if not isinstance(optimizer, dict):
            raise TypeError("training.student.optimizer is required.")
        return cls(train_type=train_type, optimizer=optimizer, lora=lora)


@dataclass(frozen=True)
class FastWAMActionConsistencyConfig:
    student: ActionStudentConfig
    target_steps: int
    teacher_reference_steps: int
    ema_decay: float
    consistency_loss_weight: float
    flow_loss_weight: float  # Weight of the selected flow/x0 supervision.
    huber_c: float
    flow_target: str = "data"
    supervision_type: str = "flow"
    teacher_start: str = "t"
    teacher_end: str = "0"
    video_conditioning: str = "observation_only"

    @classmethod
    def from_mapping(cls, config):
        training = config["training"]
        consistency = training.get("action_consistency")
        if not isinstance(consistency, dict):
            raise TypeError("training.action_consistency is required.")

        target_steps = int(consistency.get("target_steps", 2))
        teacher_reference_steps = int(consistency.get("teacher_reference_steps", 20))
        flow_target = str(consistency.get("flow_target", "data"))
        supervision_type = str(consistency.get("supervision_type", "flow"))
        teacher_start = str(consistency.get("teacher_start", "t"))
        teacher_end = str(consistency.get("teacher_end", "0"))
        video_conditioning = str(consistency.get("video_conditioning", "observation_only"))
        if video_conditioning not in {"observation_only", "one_pass_future_cache"}:
            raise ValueError("training.action_consistency.video_conditioning must be observation_only or one_pass_future_cache.")
        ema_decay = float(consistency.get("ema_decay", 0.995))
        consistency_weight = float(consistency.get("consistency_loss_weight", 1.0))
        flow_weight = float(consistency.get("flow_loss_weight", 0.2))
        huber_c = float(consistency.get("huber_c", 0.001))
        if target_steps <= 0:
            raise ValueError("training.action_consistency.target_steps must be positive.")
        if teacher_reference_steps <= 0:
            raise ValueError("training.action_consistency.teacher_reference_steps must be positive.")
        if flow_target not in {"data", "teacher"}:
            raise ValueError("training.action_consistency.flow_target must be 'data' or 'teacher'.")
        if supervision_type not in {"flow", "x0"}:
            raise ValueError("training.action_consistency.supervision_type must be 'flow' or 'x0'.")
        if (teacher_start, teacher_end) not in {("t", "0"), ("1", "0"), ("t", "r")}:
            raise ValueError("training.action_consistency.teacher_start/teacher_end must select t->0, 1->0, or t->r.")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("training.action_consistency.ema_decay must be in [0, 1).")
        if consistency_weight < 0.0 or flow_weight < 0.0 or consistency_weight + flow_weight == 0.0:
            raise ValueError("Consistency/flow loss weights must be non-negative and not both zero.")
        if huber_c <= 0.0:
            raise ValueError("training.action_consistency.huber_c must be positive.")

        return cls(
            student=ActionStudentConfig.from_mapping(training["student"]),
            target_steps=target_steps,
            teacher_reference_steps=teacher_reference_steps,
            ema_decay=ema_decay,
            consistency_loss_weight=consistency_weight,
            flow_loss_weight=flow_weight,
            huber_c=huber_c,
            flow_target=flow_target,
            supervision_type=supervision_type,
            teacher_start=teacher_start,
            teacher_end=teacher_end,
            video_conditioning=video_conditioning,
        )
