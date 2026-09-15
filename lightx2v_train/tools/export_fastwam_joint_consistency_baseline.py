"""Merge joint standard-consistency EMA/student action and video adapters."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

from export_fastwam_joint_consistency import validate_saved


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Defaults to CHECKPOINT/config.yaml.')
    parser.add_argument('--weights', choices=('ema', 'student'), default='ema')
    parser.add_argument('--train-root', type=Path, help='Directory containing the lightx2v_train package.')
    return parser.parse_args()


def resolve_train_root(checkpoint, explicit=None):
    candidates = [explicit.resolve()] if explicit else list(checkpoint.resolve().parents)
    if explicit is None:
        candidates.append(Path(__file__).resolve().parents[1])
    marker = Path('lightx2v_train/trainers/fastwam_joint_consistency_baseline/roles.py')
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError('Cannot locate the joint baseline trainer. Pass --train-root.')


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    step_text = checkpoint.name.removeprefix('checkpoint-')
    if not checkpoint.name.startswith('checkpoint-') or not step_text.isdigit():
        raise ValueError('Checkpoint directory must be named checkpoint-NNNNNNNNN.')
    step = int(step_text)
    train_root = resolve_train_root(checkpoint, args.train_root)
    sys.path[:0] = [str(train_root), str(train_root.parent)]

    import torch
    from lightx2v_train.model_zoo import build_model
    from lightx2v_train.runtime import load_config
    from lightx2v_train.trainers.fastwam_joint_consistency_baseline.config import FastWAMJointConsistencyBaselineConfig
    from lightx2v_train.trainers.fastwam_joint_consistency_baseline.roles import configure_student, load_role_state_dict

    checkpoint_config = load_config(str(checkpoint / 'config.yaml'))
    config = load_config(str(args.config)) if args.config else checkpoint_config
    for key in ('model', 'training', 'data'):
        if config[key] != checkpoint_config[key]:
            raise ValueError(f'Config and checkpoint disagree about {key}. Use the checkpoint config.')
    if config['training']['method'] != 'fastwam_joint_consistency_baseline':
        raise ValueError('This exporter requires training.method=fastwam_joint_consistency_baseline.')
    parsed = FastWAMJointConsistencyBaselineConfig.from_mapping(config)
    if not parsed.train_video:
        raise ValueError('This exporter requires training.train_video=true.')
    for role in ('action', 'video'):
        source = checkpoint / f'{args.weights}_{role}.pt'
        if not source.is_file():
            raise FileNotFoundError(source)

    print(f'Trainer: {train_root}', flush=True)
    print(f'Exporting {args.weights} action + video; step={step}; sampler=consistency_baseline; sigma_data={parsed.sigma_data}', flush=True)
    with torch.no_grad():
        model = build_model(config)
        model.load_components()
        module = model.unwrap_module()
        for role in ('action', 'video'):
            expert = configure_student(getattr(module, f'{role}_expert'), parsed.student)
            state = torch.load(checkpoint / f'{args.weights}_{role}.pt', map_location='cpu', weights_only=True)
            load_role_state_dict(expert, parsed.student.train_type, state)
            print(f'{role}: validated {len(state)} adapter/state tensors', flush=True)
            if parsed.student.train_type == 'lora':
                expert = expert.merge_and_unload(safe_merge=True)
            setattr(module, f'{role}_expert', expert)
            module.mot.mixtures[role] = expert
            del state

        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
        os.close(fd)
        try:
            module.save_checkpoint(temporary, step=step)
            saved = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
            validate_saved(torch, saved, module, step)
            os.link(temporary, output)
        finally:
            os.unlink(temporary)
    print(f'EXPORT_OK path={output} bytes={output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
