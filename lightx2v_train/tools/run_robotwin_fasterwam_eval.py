"""Run the upstream manager with explicit logical-to-physical GPU mapping.

The upstream single-task launcher overwrites CUDA_VISIBLE_DEVICES with gpu_id,
so restricting only the manager's environment does not isolate evaluations.
No upstream files are changed by this adapter.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import yaml


def map_worker_gpu(command, single_entry, visible_devices):
    if not isinstance(command, (list, tuple)) or len(command) < 2 or command[1] != str(single_entry):
        return command
    result = list(command)
    for index, argument in enumerate(result):
        if argument.startswith('gpu_id='):
            logical = int(argument.split('=', 1)[1])
            if not 0 <= logical < len(visible_devices):
                raise ValueError(f'Worker GPU {logical} exceeds configured devices {visible_devices}')
            result[index] = f'gpu_id={visible_devices[logical]}'
            return result
    raise ValueError('RoboTwin worker command is missing gpu_id')


def completed_phase_file(manager, command):
    """Reuse only a finished phase with matching checkpoint and evaluation settings."""
    values = dict(item.split('=', 1) for item in command[2:] if '=' in item)
    ckpt = manager._resolve_path(values['ckpt'], base=manager.PROJECT_ROOT)
    output = manager._resolve_path(values['EVALUATION.output_dir'], base=manager.PROJECT_ROOT)
    task = values['EVALUATION.task_name']
    phase = {'demo_clean': 'clean', 'demo_randomized': 'random'}[values['EVALUATION.task_config']]
    directory = manager.PROJECT_ROOT / 'evaluate_results/robotwin' / manager._resolve_ckpt_tag(ckpt) / output.name
    result = directory / task / manager._phase_result_filename(phase)
    config_path = directory / f'eval_config_{task}.yaml'
    if not result.is_file() or not config_path.is_file():
        return None
    saved = yaml.safe_load(config_path.read_text())
    if manager._resolve_path(saved['ckpt'], base=manager.PROJECT_ROOT) != ckpt:
        raise ValueError(f'Completed result uses a different checkpoint: {result}')
    for key in ('seed', 'EVALUATION.eval_num_episodes', 'EVALUATION.num_inference_steps',
                'EVALUATION.sigma_shift', 'EVALUATION.replan_steps',
                'EVALUATION.instruction_type', 'EVALUATION.dataset_stats_path'):
        current = saved
        for part in key.split('.'):
            current = current[part]
        if key not in values or str(current) != str(values[key]):
            raise ValueError(f'Cannot reuse {result}: evaluation setting {key} differs')
    rate = manager._parse_success_rate(result)
    if not 0 <= rate <= 1:
        raise ValueError(f'Invalid completed success rate: {result}')
    return result


def main():
    root = Path(os.environ['FASTWAM_ROOT']).resolve()
    source = root / 'experiments/robotwin/run_robotwin_manager.py'
    spec = importlib.util.spec_from_file_location('fasterwam_upstream_eval_manager', source)
    manager = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = manager
    spec.loader.exec_module(manager)
    runtime_root = os.environ.get('EVAL_RUNTIME_ROOT')
    if runtime_root:
        # Keep upstream code/task discovery, but place outputs on a writable disk.
        manager.PROJECT_ROOT = Path(runtime_root).resolve(strict=True)
    devices = os.environ.get('EVAL_CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')
    if not devices or any(not device.strip().isdigit() for device in devices):
        raise ValueError('EVAL_CUDA_VISIBLE_DEVICES must be a comma-separated list of GPU indices')
    devices = [device.strip() for device in devices]

    def launch(command, *args, **kwargs):
        mapped = map_worker_gpu(command, manager.SINGLE_ENTRY, devices)
        if os.environ.get('EVAL_RESUME_COMPLETED') == '1':
            result = completed_phase_file(manager, mapped)
            if result is not None:
                print(f'REUSE_COMPLETED_PHASE {result}', flush=True)
                # The upstream manager reads the preserved result after exit 0.
                return subprocess.Popen([sys.executable, '-c', 'pass'], *args, **kwargs)
        if runtime_root:
            mapped = list(mapped)
            mapped[1] = str(Path(__file__).with_name('run_robotwin_fasterwam_single.py'))
        return subprocess.Popen(mapped, *args, **kwargs)

    manager.subprocess = SimpleNamespace(Popen=launch, TimeoutExpired=subprocess.TimeoutExpired)
    print(f'FasterWAM evaluation logical-to-physical GPU mapping: {devices}', flush=True)
    # A dynamically imported Hydra entry cannot resolve its original relative
    # config path as a package; retain the same configs via an absolute path.
    import hydra
    hydra.main(version_base='1.3', config_path=str(root / 'configs'),
               config_name='sim_robotwin.yaml')(manager.main.__wrapped__)()


if __name__ == '__main__':
    main()
