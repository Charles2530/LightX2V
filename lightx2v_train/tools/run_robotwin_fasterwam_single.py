"""Use the unchanged upstream worker with an isolated writable runtime root.

The runtime's configs/ and experiments/ are symlinks to upstream read-only code.
Only evaluation logs/results move; model and simulator execution are unchanged.
"""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    root = Path(os.environ['FASTWAM_ROOT']).resolve(strict=True)
    runtime = Path(os.environ['EVAL_RUNTIME_ROOT']).resolve(strict=True)
    for name in ('configs', 'experiments'):
        if (runtime / name).resolve(strict=True) != (root / name).resolve(strict=True):
            raise ValueError(f'Runtime {name} must reference the unchanged upstream directory')
    spec = importlib.util.spec_from_file_location(
        'fasterwam_upstream_single', root / 'experiments/robotwin/eval_robotwin_single.py')
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    worker.PROJECT_ROOT = runtime
    import hydra
    hydra.main(version_base='1.3', config_path=str(root / 'configs'),
               config_name='sim_robotwin.yaml')(worker.main.__wrapped__)()


if __name__ == '__main__':
    main()
