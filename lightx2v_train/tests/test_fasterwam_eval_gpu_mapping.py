import importlib.util
from pathlib import Path

import pytest
import yaml
from types import SimpleNamespace


def mapper():
    path = Path(__file__).parents[1] / 'tools/run_robotwin_fasterwam_eval.py'
    spec = importlib.util.spec_from_file_location('eval_gpu_mapping_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.map_worker_gpu


@pytest.mark.parametrize('logical,physical', [(0, '2'), (1, '5')])
def test_worker_uses_physical_device_without_mutating_original(logical, physical):
    command = ['python', '/upstream/single.py', f'gpu_id={logical}', 'seed=42']
    result = mapper()(command, Path('/upstream/single.py'), ['2', '5'])
    assert result[2] == f'gpu_id={physical}'
    assert command[2] == f'gpu_id={logical}'
    assert result[-1] == 'seed=42'


def test_unrelated_process_is_unchanged():
    command = ['python', '/different.py', 'gpu_id=0']
    assert mapper()(command, Path('/upstream/single.py'), ['5']) is command


@pytest.mark.parametrize('argument', ['gpu_id=1', 'gpu_id=-1', 'seed=42'])
def test_invalid_worker_mapping_fails(argument):
    with pytest.raises(ValueError):
        mapper()(['python', '/upstream/single.py', argument], Path('/upstream/single.py'), ['0'])


def test_resume_only_reuses_matching_completed_phases(tmp_path):
    path = Path(__file__).parents[1] / 'tools/run_robotwin_fasterwam_eval.py'
    spec = importlib.util.spec_from_file_location('eval_resume_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / 'export.pt'
    evaluation = dict(eval_num_episodes=100, num_inference_steps=1, sigma_shift=5.0,
                      replan_steps=28, instruction_type='unseen', dataset_stats_path='/stats.json')
    command = ['python', '/single.py', f'ckpt={checkpoint}',
               'EVALUATION.output_dir=/run', 'EVALUATION.task_name=adjust_bottle',
               'EVALUATION.task_config=demo_clean', 'seed=42']
    command += [f'EVALUATION.{key}={value}' for key, value in evaluation.items()]
    manager = SimpleNamespace(PROJECT_ROOT=tmp_path,
        _resolve_path=lambda value, base: Path(value).resolve(),
        _resolve_ckpt_tag=lambda value: 'export',
        _phase_result_filename=lambda phase: f'_result_{phase}.txt',
        _parse_success_rate=lambda path: float(path.read_text()))
    assert module.completed_phase_file(manager, command) is None
    directory = tmp_path / 'evaluate_results/robotwin/export/run'
    (directory / 'adjust_bottle').mkdir(parents=True)
    result = directory / 'adjust_bottle/_result_clean.txt'
    result.write_text('1.0')
    config = directory / 'eval_config_adjust_bottle.yaml'
    config.write_text(yaml.safe_dump(dict(ckpt=str(checkpoint), seed=42, EVALUATION=evaluation)))
    assert module.completed_phase_file(manager, command) == result
    with pytest.raises(ValueError, match='eval_num_episodes'):
        module.completed_phase_file(manager, command + ['EVALUATION.eval_num_episodes=50'])
    with pytest.raises(ValueError, match='seed'):
        module.completed_phase_file(manager, command + ['seed=43'])
