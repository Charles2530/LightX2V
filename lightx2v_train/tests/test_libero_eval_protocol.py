import sys
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from libero_eval_protocol import load_official_evaluation_protocol


OFFICIAL_LOOP = """
dummy = np.zeros((env_num, 7))
for _ in range(5):
    pass
while steps < cfg.eval.max_steps:
    dones[k] = dones[k] or done[k]
"""


def make_protocol_files(root, metric_source=OFFICIAL_LOOP, max_steps=600):
    metric = root / "libero" / "lifelong" / "metric.py"
    config = root / "libero" / "configs" / "eval" / "default.yaml"
    metric.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    metric.write_text(metric_source, encoding="utf-8")
    config.write_text(f"max_steps: {max_steps}\n", encoding="utf-8")
    return metric, config


def test_loads_official_rollout_contract(tmp_path):
    metric, config = make_protocol_files(tmp_path)

    protocol = load_official_evaluation_protocol(tmp_path)

    assert protocol["metric_script"] == str(metric)
    assert protocol["eval_config"] == str(config)
    assert protocol["initialization_steps"] == 5
    assert protocol["initialization_action"] == [0.0] * 7
    assert protocol["max_policy_steps"] == 600
    assert len(protocol["metric_script_sha256"]) == 64
    assert len(protocol["eval_config_sha256"]) == 64


def test_rejects_unrecognized_official_loop(tmp_path):
    make_protocol_files(tmp_path, metric_source="pass\n")

    with pytest.raises(RuntimeError, match="Unsupported LIBERO-plus official evaluation"):
        load_official_evaluation_protocol(tmp_path)


def test_rejects_nonpositive_horizon(tmp_path):
    make_protocol_files(tmp_path, max_steps=0)

    with pytest.raises(ValueError, match="Invalid official max_steps"):
        load_official_evaluation_protocol(tmp_path)
