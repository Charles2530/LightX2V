"""Merge FastWAM joint consistency action/video adapters into a native checkpoint."""

import argparse
import os
from pathlib import Path
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Defaults to CHECKPOINT/config.yaml.')
    parser.add_argument('--weights', choices=('ema', 'student'), default='ema')
    parser.add_argument(
        '--train-root', type=Path,
        help='Directory containing the lightx2v_train package; inferred from checkpoint ancestors, then the script location.',
    )
    return parser.parse_args()


def resolve_train_root(checkpoint, explicit=None):
    if explicit is not None:
        candidates = [explicit.resolve()]
    else:
        candidates = list(checkpoint.resolve().parents)
        candidates.append(Path(__file__).resolve().parents[1])
    marker = Path('lightx2v_train/trainers/fastwam_joint_consistency/roles.py')
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(
        'Cannot locate the joint consistency trainer. Pass --train-root pointing '
        'to the training repository directory containing the lightx2v_train package.'
    )


def validate_saved(torch, saved, module, step):
    expected = module.mot.state_dict()
    actual = saved['mot']
    if saved['step'] != step or set(actual) != set(expected):
        raise RuntimeError('Exported checkpoint step or tensor keys do not match.')
    if any('lora_' in key for key in actual):
        raise RuntimeError('Export still contains unmerged LoRA tensors.')
    for key, tensor in expected.items():
        if actual[key].shape != tensor.shape or actual[key].dtype != tensor.dtype:
            raise RuntimeError(f'Exported tensor shape/dtype mismatch: {key}')
    for role in ('action', 'video'):
        key = next(k for k in expected if k.startswith(f'mixtures.{role}.') and k.endswith('.q.weight'))
        if not torch.equal(actual[key], expected[key].cpu()) or not torch.isfinite(actual[key]).all():
            raise RuntimeError(f'Exported tensor verification failed: {key}')
        print(f'Verified saved tensor: {key}', flush=True)
    if module.proprio_encoder is not None:
        reference = module.proprio_encoder.state_dict()
        restored = saved['proprio_encoder']
        if set(reference) != set(restored):
            raise RuntimeError('Proprio encoder tensor keys do not match.')
        for key, tensor in reference.items():
            if not torch.equal(restored[key], tensor.cpu()):
                raise RuntimeError(f'Proprio encoder verification failed: {key}')


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
    from lightx2v_train.trainers.fastwam_joint_consistency.config import FastWAMJointConsistencyConfig
    from lightx2v_train.trainers.fastwam_joint_consistency.roles import configure_student, load_role_state_dict

    checkpoint_config = load_config(str(checkpoint / 'config.yaml'))
    config = load_config(str(args.config)) if args.config else checkpoint_config
    for key in ('model', 'training', 'data'):
        if config[key] != checkpoint_config[key]:
            raise ValueError(f'Config and checkpoint disagree about {key}. Use the checkpoint config.')
    if config['training']['method'] != 'fastwam_joint_consistency':
        raise ValueError('This exporter requires training.method=fastwam_joint_consistency.')
    parsed = FastWAMJointConsistencyConfig.from_mapping(config)
    if not parsed.train_video:
        raise ValueError('This exporter requires training.train_video=true.')
    for role in ('action', 'video'):
        source = checkpoint / f'{args.weights}_{role}.pt'
        if not source.is_file():
            raise FileNotFoundError(source)

    print(f'Trainer: {train_root}', flush=True)
    print(f'Exporting {args.weights} action + video; target_steps={parsed.target_steps}; step={step}', flush=True)
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
            # Publish without replacing another export that may have finished concurrently.
            os.link(temporary, output)
        finally:
            os.unlink(temporary)
    print(f'EXPORT_OK path={output} bytes={output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
