import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_SRC = ROOT / "lightx2v_ros" / "src" / "simulator"
if str(SIMULATOR_SRC) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SRC))

from simulator.libero_node import observer


def test_plus_plain_task_init_states_fall_back_to_direct_file(monkeypatch, tmp_path):
    init_root = tmp_path / "init_files"
    task_dir = init_root / "plain_problem"
    task_dir.mkdir(parents=True)
    init_file = task_dir / "plain.init"
    init_file.write_bytes(b"placeholder")

    task = types.SimpleNamespace(
        problem_folder="plain_problem",
        init_states_file="plain.init",
        name="plain task",
    )

    class PlainTaskSuite:
        def get_task(self, task_id):
            return task

        def get_task_init_states(self, task_id):
            raise UnboundLocalError("local variable 'init_states_path' referenced before assignment")

    fake_libero = types.ModuleType("libero.libero")
    fake_libero.get_libero_path = lambda key: str(init_root) if key == "init_states" else key
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero)
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: np.arange(8, dtype=np.float32).reshape(1, 8))

    states = observer.load_task_init_states(PlainTaskSuite(), 0)

    assert states.shape == (1, 8)
    assert states.dtype == np.float32


def test_libero_config_isolated_per_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root_a = tmp_path / "libero_a"
    root_b = tmp_path / "libero_b"
    for root in (root_a, root_b):
        (root / "libero" / "libero" / "bddl_files").mkdir(parents=True)

    observer.setup_libero_config(root_a)
    config_a = Path(observer.os.environ["LIBERO_CONFIG_PATH"])
    observer.setup_libero_config(root_b)
    config_b = Path(observer.os.environ["LIBERO_CONFIG_PATH"])

    assert config_a != config_b
    assert config_a.joinpath("config.yaml").is_file()
    assert config_b.joinpath("config.yaml").is_file()
    assert str(root_a / "libero" / "libero") in config_a.joinpath("config.yaml").read_text()
    assert str(root_b / "libero" / "libero") in config_b.joinpath("config.yaml").read_text()
