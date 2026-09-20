"""Controlled RoboTwin A/B audit; does not change the production inference path.

Run under the RoboTwin conda environment with FasterWAM/src and this tools
directory on PYTHONPATH. Each arm uses the same scenes, prompts, action seed,
normalization and replan interval; only weights and cache mode differ.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType


def create_model(compile_training_denoise=False, **kwargs):
    from fasterwam.runtime import create_fasterwam
    from fasterwam.models.wan22.fastwam import FastWAM

    if compile_training_denoise:
        raise ValueError("Training compile must be disabled for this audit")
    model = create_fasterwam(**kwargs)
    mode = os.environ["FASTERWAM_AUDIT_CACHE_MODE"]
    if mode == "observation_only":
        model.infer_action = MethodType(FastWAM.infer_action, model)
    elif mode == "one_pass_future_cache":
        model.infer_action = model.infer_action_one_pass_future_cache
    else:
        raise ValueError(mode)
    print(f"AUDIT inference={model.infer_action.__func__.__qualname__} cache={mode}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="adjust_bottle,blocks_ranking_rgb,click_alarmclock,open_laptop")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    root = Path('/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM')
    light = Path(__file__).resolve().parents[2]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tasks = args.tasks.split(',')
    released = Path('/mnt/miaohua/charles/models/fasterwam_release/robotwin/step_029355.pt')
    distilled = light / 'fasterwam_robotwin_consistency_ts10_ema_step30000.pt'
    arms = [
        ('released_future', released, 10, 'one_pass_future_cache'),
        ('released_observation', released, 10, 'observation_only'),
        ('ema_future', distilled, 1, 'one_pass_future_cache'),
        ('ema_observation', distilled, 1, 'observation_only'),
    ]
    env = os.environ.copy()
    env.update(CUDA_HOME='/usr/local/cuda', CUROBO_TORCH_COMPILE_DISABLE='1',
               HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1',
               DIFFSYNTH_SKIP_DOWNLOAD='true',
               DIFFSYNTH_MODEL_BASE_PATH='/mnt/miaohua/charles/models/fastWAM-compat')
    env['PYTHONPATH'] = ':'.join([str(Path(__file__).parent), str(root/'src'), str(root),
                                '/mnt/miaohua/charles/codes/FasterWAM/src', env.get('PYTHONPATH','')])
    gl = env['ROBOTWIN_NVIDIA_GL_ROOT']
    env['VK_ICD_FILENAMES'] = gl + '/nvidia_icd_abs.json'
    env['LD_PRELOAD'] = gl + '/libGL.so.1.7.0'
    for key in ('ROBOTWIN_FORCE_RASTER','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE'):
        env.pop(key, None)

    def run_lane(arm_index, lane):
        name, checkpoint, steps, mode = arms[arm_index]
        gpu = 2*arm_index + lane
        worker_env = dict(env, FASTERWAM_AUDIT_CACHE_MODE=mode)
        rows = []
        for task in tasks[lane::2]:
            for phase, task_config in [('clean','demo_clean'),('random','demo_randomized')]:
                run_name = out.name + '_' + name
                cmd = [sys.executable, '-u', str(root/'experiments/robotwin/eval_robotwin_single.py'),
                       'task=robotwin_fasterwam_alignment_audit', f'ckpt={checkpoint}', f'gpu_id={gpu}',
                       f'EVALUATION.task_name={task}', f'EVALUATION.task_config={task_config}',
                       f'EVALUATION.output_dir={out/run_name}', f'EVALUATION.eval_num_episodes={args.episodes}',
                       f'EVALUATION.num_inference_steps={steps}', 'EVALUATION.sigma_shift=5.0',
                       'EVALUATION.replan_steps=28', 'EVALUATION.instruction_type=unseen',
                       'EVALUATION.skip_get_obs_within_replan=true', 'EVALUATION.reuse_seed_cache=true',
                       'EVALUATION.dataset_stats_path=/mnt/miaohua/charles/models/fasterwam_release/robotwin/dataset_stats.json']
                log = out/f'{name}_{task}_{phase}.log'
                print(f'START arm={name} task={task} phase={phase} gpu={gpu}', flush=True)
                with log.open('w') as stream:
                    stream.write(json.dumps(cmd)+'\n'); stream.flush()
                    rc = subprocess.call(cmd, cwd=root, env=worker_env, stdout=stream, stderr=subprocess.STDOUT)
                result = root/'evaluate_results/robotwin'/checkpoint.stem/run_name/task/f'_result_{phase}.txt'
                rate = None
                if rc == 0 and result.is_file():
                    rate = float(result.read_text().strip().splitlines()[-1])
                row = dict(arm=name, task=task, phase=phase, gpu=gpu, return_code=rc,
                           success_rate=rate, log=str(log), result=str(result))
                rows.append(row)
                print('DONE '+json.dumps(row), flush=True)
                if rc or rate is None:
                    raise RuntimeError(f'Audit worker failed: {row}')
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_lane, arm, lane) for arm in range(4) for lane in range(2)]
        for future in as_completed(futures):
            rows.extend(future.result())
            (out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('ALL_DONE '+str(out/'results.json'), flush=True)


if __name__ == '__main__':
    main()
