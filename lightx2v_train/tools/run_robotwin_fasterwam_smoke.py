"""Run the real RoboTwin manager on a bounded task subset, without changing episodes."""
import importlib.util
import os
from pathlib import Path

root = Path(os.environ['FASTWAM_ROOT']).resolve()
spec = importlib.util.spec_from_file_location(
    'robotwin_smoke_manager', root / 'experiments/robotwin/run_robotwin_manager.py'
)
manager = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = manager
spec.loader.exec_module(manager)
tasks = manager._load_all_tasks()
limit = int(os.environ.get('ROBOTWIN_SMOKE_TASK_LIMIT', '16'))
if not 1 <= limit <= len(tasks):
    raise ValueError(f'Invalid smoke task limit: {limit}')
# Use the production scheduling, workers, and result collection on these tasks.
manager._load_all_tasks = lambda: tasks[:limit]
if __name__ == '__main__':
    sys.argv[1:1] = ['--config-path', str(root / 'configs')]
    manager.main()
