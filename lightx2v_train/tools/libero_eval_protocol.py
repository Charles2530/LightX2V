"""Read and validate the rollout protocol shipped by LIBERO-plus."""

import hashlib
from pathlib import Path

import yaml


OFFICIAL_METRIC_RELATIVE_PATH = Path("libero/lifelong/metric.py")
OFFICIAL_EVAL_CONFIG_RELATIVE_PATH = Path("libero/configs/eval/default.yaml")
OFFICIAL_INITIALIZATION_STEPS = 5
PROTOCOL_NAME = "libero-plus-lifelong-metric-v1"
IMPLEMENTATION_RELATIVE_PATHS = (
    Path("lightx2v_train/tools/eval_fastwam_libero.py"),
    Path("lightx2v_train/tools/libero_eval_protocol.py"),
    Path("lightx2v_ros/src/simulator/simulator/libero_node/observer.py"),
    Path("lightx2v/models/runners/wan/fastwam_runner.py"),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_evaluation_protocol(libero_root):
    libero_root = Path(libero_root).expanduser().resolve()
    metric_path = libero_root / OFFICIAL_METRIC_RELATIVE_PATH
    config_path = libero_root / OFFICIAL_EVAL_CONFIG_RELATIVE_PATH
    if not metric_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"LIBERO-plus official evaluation files are missing: {metric_path}, {config_path}"
        )

    metric_source = metric_path.read_text(encoding="utf-8")
    required_source = (
        "dummy = np.zeros((env_num, 7))",
        f"for _ in range({OFFICIAL_INITIALIZATION_STEPS}):",
        "while steps < cfg.eval.max_steps:",
        "dones[k] = dones[k] or done[k]",
    )
    missing = [snippet for snippet in required_source if snippet not in metric_source]
    if missing:
        raise RuntimeError(
            f"Unsupported LIBERO-plus official evaluation implementation at {metric_path}; "
            f"missing expected source fragments: {missing}"
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    max_steps = int(config["max_steps"])
    if max_steps <= 0:
        raise ValueError(f"Invalid official max_steps={max_steps} in {config_path}")
    return {
        "name": PROTOCOL_NAME,
        "metric_script": str(metric_path),
        "metric_script_sha256": sha256_file(metric_path),
        "eval_config": str(config_path),
        "eval_config_sha256": sha256_file(config_path),
        "initialization_steps": OFFICIAL_INITIALIZATION_STEPS,
        "initialization_action": [0.0] * 7,
        "max_policy_steps": max_steps,
        "success_rule": "latch environment done and stop after success",
        "initial_state_rule": "episode_index modulo official task initial-state count",
    }


def load_fastwam_evaluation_implementation(repo_root):
    repo_root = Path(repo_root).expanduser().resolve()
    files = {}
    for relative_path in IMPLEMENTATION_RELATIVE_PATHS:
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        files[str(relative_path)] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return files
