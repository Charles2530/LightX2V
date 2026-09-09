"""Merge FastWAM action-only TBSM LoRA weights into a native checkpoint."""

import argparse
import os
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Training checkpoint directory, e.g. checkpoint-000030000.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output native FastWAM checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Training config. Defaults to CHECKPOINT/config.yaml.",
    )
    parser.add_argument(
        "--weights",
        choices=("ema", "student"),
        default="ema",
        help="Which action weights to export.",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        help=(
            "Directory containing the lightx2v_train package. "
            "Inferred from checkpoint ancestors, then the script's training directory."
        ),
    )
    return parser.parse_args()


def resolve_train_root(checkpoint, explicit=None):
    if explicit is not None:
        candidates = [explicit.resolve()]
    else:
        candidates = list(checkpoint.resolve().parents)
        candidates.append(Path(__file__).resolve().parents[1])

    marker = Path(
        "lightx2v_train/trainers/fastwam_action_tbsm/config.py"
    )

    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate

    raise FileNotFoundError(
        "Cannot locate lightx2v_train with the TBSM trainer. "
        "Pass --train-root explicitly."
    )


def validate_saved(torch, saved, module, step):
    if saved["step"] != step:
        raise RuntimeError(
            f"Exported checkpoint step mismatch: "
            f"expected={step}, actual={saved['step']}"
        )

    expected = module.mot.state_dict()
    actual = saved["mot"]

    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "Exported checkpoint tensor keys do not match.\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    # Native checkpoint must no longer contain PEFT LoRA tensors.
    lora_keys = [key for key in actual if "lora_" in key]
    if lora_keys:
        raise RuntimeError(
            f"Export still contains unmerged LoRA tensors: "
            f"{lora_keys[:10]}"
        )

    # Check shape/dtype for every MoT tensor.
    for key, tensor in expected.items():
        if actual[key].shape != tensor.shape:
            raise RuntimeError(
                f"Exported tensor shape mismatch for {key}: "
                f"{actual[key].shape} != {tensor.shape}"
            )
        if actual[key].dtype != tensor.dtype:
            raise RuntimeError(
                f"Exported tensor dtype mismatch for {key}: "
                f"{actual[key].dtype} != {tensor.dtype}"
            )

    # Verify one actual action-expert weight bit-for-bit.
    action_keys = [
        key
        for key in expected
        if key.startswith("mixtures.action.")
        and key.endswith(".q.weight")
    ]

    if not action_keys:
        raise RuntimeError(
            "Could not find an action q.weight for export verification."
        )

    key = action_keys[0]

    if not torch.equal(actual[key], expected[key].cpu()):
        raise RuntimeError(
            f"Exported action tensor does not match merged model: {key}"
        )

    if not torch.isfinite(actual[key]).all():
        raise RuntimeError(
            f"Exported action tensor contains NaN/Inf: {key}"
        )

    print(f"Verified saved action tensor: {key}", flush=True)


def main():
    args = parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.absolute()

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint}"
        )

    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )

    # Extract training step from checkpoint-000030000.
    checkpoint_name = checkpoint.name
    prefix = "checkpoint-"

    if not checkpoint_name.startswith(prefix):
        raise ValueError(
            "Checkpoint directory must be named checkpoint-NNNNNNNNN."
        )

    step_text = checkpoint_name[len(prefix):]

    if not step_text.isdigit():
        raise ValueError(
            "Checkpoint directory must be named checkpoint-NNNNNNNNN."
        )

    step = int(step_text)

    # Locate the actual training repository so we use the exact trainer code
    # corresponding to this checkpoint.
    train_root = resolve_train_root(
        checkpoint,
        args.train_root,
    )

    sys.path[:0] = [
        str(train_root),
        str(train_root.parent),
    ]

    import torch

    from lightx2v_train.model_zoo import build_model
    from lightx2v_train.runtime import load_config
    from lightx2v_train.trainers.fastwam_action_tbsm.config import (
        FastWAMActionTBSMConfig,
    )
    from lightx2v_train.trainers.fastwam_action_consistency.roles import (
        configure_student,
        load_role_state_dict,
    )

    # Prefer the exact config saved with this checkpoint.
    checkpoint_config_path = checkpoint / "config.yaml"

    if not checkpoint_config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint config is missing: {checkpoint_config_path}"
        )

    checkpoint_config = load_config(
        str(checkpoint_config_path)
    )

    config = (
        load_config(str(args.config))
        if args.config is not None
        else checkpoint_config
    )

    # If an external config is supplied, require the model/training/data
    # sections to be identical to the saved training config.
    for key in ("model", "training", "data"):
        if config[key] != checkpoint_config[key]:
            raise ValueError(
                f"Config and checkpoint disagree about {key}. "
                "Use the checkpoint config."
            )

    if config["training"]["method"] != "fastwam_action_tbsm":
        raise ValueError(
            "This exporter requires "
            "training.method=fastwam_action_tbsm."
        )

    parsed = FastWAMActionTBSMConfig.from_mapping(config)

    source = checkpoint / f"{args.weights}_action.pt"

    if not source.is_file():
        raise FileNotFoundError(
            f"Action weight file is missing: {source}"
        )

    print(f"Trainer root : {train_root}", flush=True)
    print(f"Checkpoint   : {checkpoint}", flush=True)
    print(f"Weight type  : {args.weights}", flush=True)
    print(f"Action state : {source}", flush=True)
    print(f"Step         : {step}", flush=True)
    print(
        f"Train type   : {parsed.student.train_type}",
        flush=True,
    )

    with torch.no_grad():
        # build_model() loads the base FastWAM checkpoint specified by
        # model.checkpoint_path in the training config.
        model = build_model(config)
        model.load_components()

        module = model.unwrap_module()

        # ------------------------------------------------------------
        # Action only
        #
        # Reconstruct exactly the same LoRA architecture as training.
        # ------------------------------------------------------------
        action = configure_student(
            module.action_expert,
            parsed.student,
        )

        state = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )

        # This function strictly verifies that the saved adapter keys
        # match the LoRA architecture reconstructed from config.yaml.
        load_role_state_dict(
            action,
            parsed.student.train_type,
            state,
        )

        print(
            f"Loaded and validated {len(state)} "
            f"{args.weights} action state tensors",
            flush=True,
        )

        # Merge:
        #
        # W_native = W_base + delta_W_EMA
        #
        # After this operation the exported native checkpoint no longer
        # depends on PEFT at inference time.
        if parsed.student.train_type == "lora":
            action = action.merge_and_unload(
                safe_merge=True
            )

        # Replace only action.
        #
        # video_expert is intentionally untouched and remains exactly
        # the base FastWAM video expert loaded from model.checkpoint_path.
        module.action_expert = action
        module.mot.mixtures["action"] = action

        del state

        # ------------------------------------------------------------
        # Atomic native checkpoint export
        # ------------------------------------------------------------
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temporary = tempfile.mkstemp(
            prefix=output.name + ".",
            suffix=".tmp",
            dir=output.parent,
        )
        os.close(fd)

        try:
            module.save_checkpoint(
                temporary,
                step=step,
            )

            saved = torch.load(
                temporary,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )

            validate_saved(
                torch,
                saved,
                module,
                step,
            )

            # Do not silently replace a concurrently-created export.
            os.link(
                temporary,
                output,
            )

        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    print(
        f"EXPORT_OK path={output} "
        f"bytes={output.stat().st_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
