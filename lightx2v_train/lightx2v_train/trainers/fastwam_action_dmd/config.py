from dataclasses import dataclass


@dataclass(frozen=True)
class ActionRoleConfig:
    train_type: str
    optimizer: dict
    lora: dict | None

    @classmethod
    def from_mapping(cls, mapping, name):
        train_type = str(mapping.get("train_type", "lora")).lower()
        if train_type not in {"lora", "full"}:
            raise ValueError(f"training.{name}.train_type must be 'lora' or 'full', got {train_type!r}.")
        lora = mapping.get("lora")
        if train_type == "lora":
            if not isinstance(lora, dict):
                raise ValueError(f"training.{name}.lora is required for LoRA training.")
            if int(lora.get("rank", 0)) <= 0:
                raise ValueError(f"training.{name}.lora.rank must be positive.")
        optimizer = mapping.get("optimizer")
        if not isinstance(optimizer, dict):
            raise TypeError(f"training.{name}.optimizer is required.")
        return cls(train_type=train_type, optimizer=optimizer, lora=lora)


@dataclass(frozen=True)
class FastWAMActionDmdConfig:
    student: ActionRoleConfig
    fake: ActionRoleConfig
    teacher_steps: int
    endpoint_warmup_iters: int
    endpoint_loss_weight: float
    dmd_loss_weight: float
    fake_loss_weight: float
    fake_update_ratio: int
    sigma_min: float
    sigma_max: float
    norm_clip_min: float | None

    @classmethod
    def from_mapping(cls, config):
        training = config["training"]
        dmd = training.get("action_dmd")
        if not isinstance(dmd, dict):
            raise TypeError("training.action_dmd is required.")
        if int(dmd.get("generator_steps", 1)) != 1:
            raise ValueError("FastWAM action DMD currently supports generator_steps=1 only.")
        sigma_min = float(dmd.get("sigma_min", 0.02))
        sigma_max = float(dmd.get("sigma_max", 0.98))
        if not 0.0 < sigma_min < sigma_max < 1.0:
            raise ValueError(f"Expected 0 < sigma_min < sigma_max < 1, got {sigma_min}, {sigma_max}.")
        teacher_steps = int(dmd.get("teacher_steps", 20))
        if teacher_steps <= 0:
            raise ValueError("training.action_dmd.teacher_steps must be positive.")
        return cls(
            student=ActionRoleConfig.from_mapping(training["student"], "student"),
            fake=ActionRoleConfig.from_mapping(training["fake"], "fake"),
            teacher_steps=teacher_steps,
            endpoint_warmup_iters=max(0, int(dmd.get("endpoint_warmup_iters", 0))),
            endpoint_loss_weight=float(dmd.get("endpoint_loss_weight", 0.0)),
            dmd_loss_weight=float(dmd.get("dmd_loss_weight", 1.0)),
            fake_loss_weight=float(dmd.get("fake_loss_weight", 1.0)),
            fake_update_ratio=max(1, int(dmd.get("fake_update_ratio", 1))),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            norm_clip_min=None if dmd.get("norm_clip_min") is None else float(dmd["norm_clip_min"]),
        )
