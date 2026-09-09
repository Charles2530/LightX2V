import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOLS = (
    ('export_fastwam_action_tbsm', 'fastwam_action_tbsm/config.py'),
    ('export_fastwam_joint_consistency', 'fastwam_joint_consistency/roles.py'),
)
TRAIN_ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TRAIN_ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportToolPathsTest(unittest.TestCase):
    def test_trainer_resolution(self):
        for name, marker in TOOLS:
            with self.subTest(tool=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint_root = root / 'training'
                local_root = root / 'local_training'
                explicit_root = root / 'explicit_training'
                for candidate in (checkpoint_root, local_root, explicit_root):
                    file = candidate / 'lightx2v_train' / 'trainers' / marker
                    file.parent.mkdir(parents=True)
                    file.touch()
                checkpoint = checkpoint_root / 'runs' / 'run' / 'checkpoint-000030000'
                detached = root / 'detached' / 'checkpoint-000030000'
                module = load_tool(name)
                with patch.object(module, '__file__', str(local_root / 'tools' / f'{name}.py')):
                    self.assertEqual(module.resolve_train_root(checkpoint), checkpoint_root)
                    self.assertEqual(module.resolve_train_root(detached), local_root)
                    self.assertEqual(module.resolve_train_root(checkpoint, explicit_root), explicit_root)
                    with self.assertRaises(FileNotFoundError):
                        module.resolve_train_root(checkpoint, root / 'missing')
                with patch.object(module, '__file__', str(root / 'missing' / 'tools' / f'{name}.py')):
                    with self.assertRaises(FileNotFoundError):
                        module.resolve_train_root(detached)

    def test_help_from_any_working_directory(self):
        for name, _ in TOOLS:
            for cwd in (TRAIN_ROOT.parent, TRAIN_ROOT, Path('/tmp')):
                with self.subTest(tool=name, cwd=cwd):
                    result = subprocess.run(
                        [sys.executable, str(TRAIN_ROOT / 'tools' / f'{name}.py'), '--help'],
                        cwd=cwd, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--checkpoint', result.stdout)
                    self.assertIn('--train-root', result.stdout)


if __name__ == '__main__':
    unittest.main()
