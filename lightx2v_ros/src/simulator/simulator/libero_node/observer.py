import hashlib
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

LIBERO_BENCHMARKS = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def default_libero_root():
    return Path(__file__).resolve().parent / "LIBERO"


def add_python_path(path):
    path = str(Path(path).expanduser())
    if path not in sys.path:
        sys.path.insert(0, path)


def setup_libero_config(libero_root):
    libero_root = Path(libero_root).expanduser().resolve()
    benchmark_root = libero_root / "libero" / "libero"
    if not (benchmark_root / "bddl_files").exists():
        raise FileNotFoundError(f"LIBERO submodule is incomplete: {libero_root}")

    root_digest = hashlib.sha256(os.fsencode(str(libero_root))).hexdigest()[:16]
    config_dir = Path.home() / ".cache" / "lightx2v_ros" / "libero_config" / root_digest
    config_file = config_dir / "config.yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(
        [
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {libero_root / 'libero' / 'datasets'}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        ]
    )
    temporary = config_file.with_name(f".{config_file.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, config_file)
    finally:
        temporary.unlink(missing_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def load_libero(libero_root):
    add_python_path(libero_root)
    setup_libero_config(libero_root)

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as exc:
        if exc.name in {"robosuite", "bddl"}:
            raise ModuleNotFoundError(f"Missing dependency '{exc.name}'. Activate the LIBERO runtime first.") from exc
        raise

    return benchmark, get_libero_path, OffScreenRenderEnv


def load_init_states(get_libero_path, task, init_state_id):
    state, _ = load_init_state(get_libero_path, task, init_state_id)
    return state


def load_init_state(get_libero_path, task, init_state_id):
    import torch

    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    index = int(init_state_id)
    if index < 0 or index >= len(init_states):
        raise ValueError(f"init_state_id {index} is out of range for {task.name!r}; expected 0..{len(init_states) - 1}")
    return init_states[index], len(init_states)


def load_task_init_states(task_suite, task_id):
    """Load states through the benchmark so LIBERO-plus can resolve perturbation tasks."""
    import torch

    torch_load = torch.load

    def load_trusted_init_states(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    # LIBERO-plus predates the PyTorch 2.6 weights_only default change.
    torch.load = load_trusted_init_states
    try:
        try:
            init_states = task_suite.get_task_init_states(int(task_id))
        except UnboundLocalError as exc:
            # Some LIBERO-plus releases omit the fallback branch for ordinary
            # LIBERO-90 task filenames.  Their direct file layout is still
            # valid, so recover through the canonical LIBERO path API.
            if "init_states_path" not in str(exc):
                raise
            from libero.libero import get_libero_path

            task = task_suite.get_task(int(task_id))
            init_states_path = (
                Path(get_libero_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            )
            init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    finally:
        torch.load = torch_load
    if hasattr(init_states, "detach"):
        init_states = init_states.detach().cpu().numpy()
    init_states = np.asarray(init_states)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)
    if len(init_states) == 0:
        raise ValueError(f"Task {task_id} has no initial states")
    return init_states


def create_task_suite(factory):
    # LIBERO-plus prints thousands of task ids every time a suite is instantiated.
    with redirect_stdout(StringIO()):
        return factory()


def build_task_catalog(benchmark_module):
    """Return stable UI task ids mapped to their LIBERO suite/task metadata."""
    factories = benchmark_module.get_benchmark_dict()
    catalog = {}
    for benchmark_name in LIBERO_BENCHMARKS:
        factory = factories.get(benchmark_name)
        if factory is None:
            continue
        task_suite = create_task_suite(factory)
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            key = f"{benchmark_name}/{task_id}"
            catalog[key] = {
                "benchmark": benchmark_name,
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
            }
    return catalog


class LiberoActionObserver:
    def __init__(
        self,
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        image_size=224,
        seed=0,
        libero_root=None,
    ):
        self.libero_root = Path(libero_root or default_libero_root()).expanduser()
        benchmark, get_libero_path, env_cls = load_libero(self.libero_root)

        self.benchmark_module = benchmark
        self.benchmark_name = str(benchmark_name).strip().lower()
        factories = benchmark.get_benchmark_dict()
        if self.benchmark_name not in factories or self.benchmark_name not in LIBERO_BENCHMARKS:
            raise ValueError(f"unknown LIBERO benchmark {benchmark_name!r}; available: {', '.join(LIBERO_BENCHMARKS)}")
        task_suite = create_task_suite(factories[self.benchmark_name])
        self.task_suite = task_suite
        self.task_id = int(task_id)
        if self.task_id < 0 or self.task_id >= task_suite.get_num_tasks():
            raise ValueError(f"task_id {self.task_id} is out of range for {self.benchmark_name!r}; expected 0..{task_suite.get_num_tasks() - 1}")
        task = task_suite.get_task(self.task_id)
        self.task = task
        self.task_description = task.language
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        # Keep an owned copy: every restart must restore this exact MuJoCo state,
        # rather than returning the observation cached after the last action.
        self.init_states = load_task_init_states(task_suite, self.task_id)
        self.num_init_states = len(self.init_states)
        self.set_init_state_id(init_state_id)
        self.image_size = int(image_size)
        self.seed = int(seed)

        self.env = env_cls(
            bddl_file_name=str(bddl_file),
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            camera_names=["robot0_eye_in_hand", "agentview", "frontview", "galleryview"],
        )
        self.env.seed(self.seed)
        self.reset()

    @property
    def task_key(self):
        return f"{self.benchmark_name}/{self.task_id}"

    def set_init_state_id(self, init_state_id):
        index = int(init_state_id)
        if index < 0 or index >= self.num_init_states:
            raise ValueError(
                f"init_state_id {index} is out of range for {self.task.name!r}; "
                f"expected 0..{self.num_init_states - 1}"
            )
        self.init_state_id = index
        self.init_state = self.init_states[index].copy()

    def reset(self):
        """Reset simulator internals and restore the configured initial state."""
        self.env.reset()
        self.obs = self.env.set_init_state(self.init_state.copy())
        return self.obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.obs, reward, success, info = self.env.step(action)
        return self.obs, reward, success, info

    def close(self):
        self.env.close()
