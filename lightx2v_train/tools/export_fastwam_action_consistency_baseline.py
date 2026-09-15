"""Export standard-consistency action adapters without changing the existing exporter."""

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
    marker = Path('lightx2v_train/trainers/fastwam_action_consistency_baseline/config.py')
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError('Cannot locate the baseline trainer. Pass --train-root.')


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
    from lightx2v_train.trainers.fastwam_action_consistency_baseline.config import FastWAMActionConsistencyBaselineConfig
    from lightx2v_train.trainers.fastwam_action_dmd.checkpoint import load_role_state_dict
    from lightx2v_train.trainers.fastwam_action_dmd.roles import configure_action_role

    checkpoint_config = load_config(str(checkpoint / 'config.yaml'))
    config = load_config(str(args.config)) if args.config else checkpoint_config
    for key in ('model', 'training', 'data'):
        if config[key] != checkpoint_config[key]:
            raise ValueError(f'Config and checkpoint disagree about {key}. Use the checkpoint config.')
    if config['training']['method'] != 'fastwam_action_consistency_baseline':
        raise ValueError('This exporter requires training.method=fastwam_action_consistency_baseline.')
    parsed = FastWAMActionConsistencyBaselineConfig.from_mapping(config)
    source = checkpoint / f'{args.weights}_action.pt'
    if not source.is_file():
        raise FileNotFoundError(source)

    print(f'Trainer: {train_root}', flush=True)
    print(f'Exporting {args.weights} action; step={step}; sampler=consistency_baseline; sigma_data={parsed.sigma_data}', flush=True)
    with torch.no_grad():
        model = build_model(config)
        model.load_components()
        module = model.unwrap_module()
        action = configure_action_role(module.action_expert, parsed.student)
        state = torch.load(source, map_location='cpu', weights_only=True)
        load_role_state_dict(action, parsed.student.train_type, state)
        print(f'action: validated {len(state)} adapter/state tensors', flush=True)
        if parsed.student.train_type == 'lora':
            action = action.merge_and_unload(safe_merge=True)
        module.action_expert = action
        module.mot.mixtures['action'] = action
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
