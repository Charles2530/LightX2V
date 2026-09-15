import copy
import csv
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from lightx2v_train.trainers.fastwam_action_dmd.lora_monitor import LoraChangeMonitor, LoraMonitorConfig
from lightx2v_train.trainers.fastwam_action_dmd.trainer import FastWAMActionDmdTrainer
from peft import LoraConfig, get_peft_model
from torch import nn


def _expert():
    expert = get_peft_model(nn.Sequential(nn.Linear(5, 4, bias=False)), LoraConfig(r=2, lora_alpha=3, target_modules=["0"]))
    layer = expert.base_model.model[0]
    return expert, layer


def _rows(monitor):
    with (monitor.directory / "layers.csv").open() as handle:
        return list(csv.DictReader(handle))


def _dense(layer):
    return layer.scaling["default"] * layer.lora_B["default"].weight.detach().double() @ layer.lora_A["default"].weight.detach().double()


def test_effective_delta_and_interval_directions_match_dense_calculation(tmp_path):
    expert, layer = _expert()
    monitor = LoraChangeMonitor(LoraMonitorConfig(every_n_steps=1), {"action": expert, "video": None}, tmp_path)
    history = [_dense(layer)]
    direction = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
    for step, sign in [(1, 1), (2, 1), (3, -1)]:
        with torch.no_grad():
            layer.lora_B["default"].weight.add_(direction * sign)
        current = _dense(layer)
        scalars = monitor.record(step)
        row = _rows(monitor)[-1]
        assert float(row["delta_w_norm"]) == pytest.approx(float(current.norm()), rel=1e-10)
        assert float(row["relative_delta_w_norm"]) == pytest.approx(float(current.norm() / layer.base_layer.weight.double().norm()), rel=1e-10)
        assert float(row["interval_update_norm"]) == pytest.approx(float((current - history[-1]).norm()), rel=1e-9)
        if step == 1:
            assert row["update_cosine"] == ""
        else:
            u, v = current - history[-1], history[-1] - history[-2]
            expected = float((u * v).sum() / (u.norm() * v.norm()))
            assert float(row["update_cosine"]) == pytest.approx(expected, abs=1e-10)
            assert float(scalars["lora_monitor/action/reversal_fraction"]) == (step == 3)
        history.append(current)
    assert float(row["reversal_fraction_so_far"]) == 0.5
    assert len(list(monitor.directory.glob("step-*.pt"))) == 2
    saved = torch.load(monitor.directory / "step-000000003.pt", weights_only=True)
    factors = next(iter(saved["layers"].values()))
    torch.testing.assert_close(factors["scale"] * factors["b"].double() @ factors["a"].double(), history[-1])
    assert scalars["lora_monitor/video/has_lora"] == 0
    assert "lora_monitor/video/delta_w_norm_mean" not in scalars


def test_factor_reparameterization_is_not_an_effective_weight_update(tmp_path):
    expert, layer = _expert()
    with torch.no_grad():
        layer.lora_B["default"].weight.fill_(0.5)
    monitor = LoraChangeMonitor(LoraMonitorConfig(every_n_steps=1), {"action": expert}, tmp_path)
    with torch.no_grad():
        layer.lora_A["default"].weight.mul_(2)
        layer.lora_B["default"].weight.mul_(0.5)
    metrics = monitor.record(1)
    assert metrics["lora_monitor/action/interval_update_norm_max"] < 1e-12
    assert metrics["lora_monitor/action/unchanged_fraction"] == 1


def test_small_effective_changes_match_dense_difference_and_scale_changes(tmp_path):
    expert, layer = _expert()
    with torch.no_grad():
        layer.lora_B["default"].weight.fill_(1)
    monitor = LoraChangeMonitor(LoraMonitorConfig(every_n_steps=1), {"action": expert}, tmp_path)
    previous = _dense(layer)
    with torch.no_grad():
        layer.lora_A["default"].weight.add_(1e-6)
        layer.lora_B["default"].weight.add_(2e-6)
    metrics = monitor.record(1)
    assert metrics["lora_monitor/action/interval_update_norm_mean"] == pytest.approx(float((_dense(layer) - previous).norm()), rel=1e-6)
    previous = _dense(layer)
    layer.scaling["default"] *= 0.75
    metrics = monitor.record(2)
    assert metrics["lora_monitor/action/interval_update_norm_mean"] == pytest.approx(float((_dense(layer) - previous).norm()), rel=1e-10)


def test_zero_updates_missing_grads_and_resume_start_have_no_fake_cosine(tmp_path):
    expert, _ = _expert()
    monitor = LoraChangeMonitor(LoraMonitorConfig(every_n_steps=10, save_snapshots=False), {"action": expert}, tmp_path, initial_step=53)
    assert not monitor.due(53)
    assert not monitor.due(59)
    assert monitor.record(59) == {}
    for step in [60, 70]:
        monitor.capture_gradients("action", step)
        metrics = monitor.record(step)
        assert metrics["lora_monitor/action/valid_direction_count"] == 0
        assert "lora_monitor/action/reversal_fraction" not in metrics
        assert metrics["lora_monitor/action/missing_gradient_fraction"] == 1
        assert metrics["lora_monitor/action/gradient_norm_mean"] == 0
    assert _rows(monitor)[1]["previous_step"] == "53"
    assert not list(monitor.directory.glob("*.pt"))


def test_nonfinite_values_are_flagged_instead_of_silently_averaged(tmp_path):
    expert, layer = _expert()
    monitor = LoraChangeMonitor(LoraMonitorConfig(every_n_steps=1), {"action": expert}, tmp_path)
    with torch.no_grad():
        layer.lora_B["default"].weight.fill_(float("nan"))
    layer.lora_B["default"].weight.grad = torch.full_like(layer.lora_B["default"].weight, float("nan"))
    monitor.capture_gradients("action", 1)
    metrics = monitor.record(1)
    assert metrics["lora_monitor/action/nonfinite_weight_layer_count"] == 1
    assert metrics["lora_monitor/action/nonfinite_gradient_layer_count"] == 1
    assert _rows(monitor)[-1]["delta_w_norm"] == ""


@pytest.mark.parametrize("mapping", [{"every_n_steps": 0}, {"every_n_steps": 1.5}, {"reversal_cos_threshold": 0.5}, {"zero_tolerance": -1}, {"keep_last_snapshots": -1}])
def test_monitor_config_validation(mapping):
    with pytest.raises(ValueError):
        LoraMonitorConfig.from_mapping(mapping)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_expert = nn.Sequential(nn.Linear(5, 4, bias=False))
        self.video_expert = nn.Sequential(nn.Linear(5, 4, bias=False))
        self.mot = nn.Module()
        self.mot.mixtures = nn.ModuleDict({"video": self.video_expert})


class _Metrics:
    def __init__(self):
        self.logs = []

    def log_metrics(self, metrics, step=None):
        self.logs.append((step, metrics))

    def finish(self):
        pass


def _run_tiny_trainer(path, enabled, video_enabled=True):
    torch.manual_seed(811)
    role = {"train_type": "lora", "lora": {"rank": 2, "alpha": 3, "target_modules": ["0"]}, "optimizer": {"learning_rate": 0.01}}
    config = {
        "training": {
            "output_dir": str(path),
            "max_train_iters": 3,
            "gradient_accumulation_iters": 2,
            "max_grad_norm": 0.001,
            "save_final": False,
            "student": copy.deepcopy(role),
            "fake": copy.deepcopy(role),
            "video": copy.deepcopy(role),
            "unfreeze_video": video_enabled,
            "action_dmd": {"fake_update_ratio": 2},
        },
        "data": {"train": {"batch_size": 1}},
        "logging": {"train_log_every_iters": 3, "lora_monitor": {"enabled": enabled, "every_n_steps": 1}},
    }
    trainer = FastWAMActionDmdTrainer(config)
    model = _TinyModel()
    trainer.set_model(SimpleNamespace(unwrap_module=lambda: model, autocast_context=nullcontext))
    trainer.monitor = _Metrics()
    trainer._iter_train_samples = lambda: iter([torch.ones(1, 5)] * 30)
    trainer._prepare_batch = lambda sample, **kwargs: ({"x": sample}, None, None)
    expected_gradients = {}

    def student_loss(inputs, condition, valid_mask, current_iter):
        prediction = trainer.roles.student(inputs["x"])
        if trainer.video_expert is not None:
            prediction = prediction + trainer.video_expert(inputs["x"])
        loss = (prediction - 2).square().mean()
        return loss, prediction.detach(), {"endpoint": loss.detach(), "dmd": loss.detach(), "video_anchor": loss.detach() * 0}

    trainer._student_loss = student_loss
    original_setup = trainer.setup

    def setup():
        step = original_setup()
        if trainer.lora_monitor is not None:
            original_capture = trainer.lora_monitor.capture_gradients

            def capture(branch, iteration):
                params = trainer.student_params if branch == "action" else trainer.video_params
                expected_gradients[(iteration, branch)] = sum(float(p.grad.double().square().sum()) for p in params if p.grad is not None) ** 0.5
                original_capture(branch, iteration)

            trainer.lora_monitor.capture_gradients = capture
        return step

    trainer.setup = setup

    def train_fake(samples):
        # Exercise the actual fake optimizer after monitored gradients have
        # been cleared. Fake gradients must never leak into student statistics.
        for _ in range(2):
            trainer.fake_optimizer.zero_grad(set_to_none=True)
            loss = trainer.roles.fake(next(samples)).square().mean()
            loss.backward()
            trainer.fake_optimizer.step()
            trainer.fake_scheduler.step()
            trainer.fake_optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()), 0.0

    trainer._train_fake_updates = train_fake
    trainer.train()
    return trainer, expected_gradients


@pytest.mark.parametrize("video_enabled", [False, True])
def test_training_and_rng_identical_with_monitor_on_and_off(tmp_path, video_enabled):
    plain, _ = _run_tiny_trainer(tmp_path / "off", False, video_enabled)
    plain_rng = torch.get_rng_state().clone()
    observed, expected = _run_tiny_trainer(tmp_path / "on", True, video_enabled)
    assert torch.equal(torch.get_rng_state(), plain_rng)
    for left, right in [(plain.roles.student, observed.roles.student), (plain.roles.fake, observed.roles.fake), (plain.video_expert, observed.video_expert)]:
        if left is not None:
            for key, value in left.state_dict().items():
                assert torch.equal(value, right.state_dict()[key]), key
    for row in _rows(observed.lora_monitor):
        if int(row["step"]) > 0:
            assert float(row["gradient_norm"]) == pytest.approx(expected[(int(row["step"]), row["branch"])])
            assert float(row["gradient_norm"]) > observed.max_grad_norm
    assert [step for step, _ in observed.monitor.logs] == [1, 2, 3]
    for step, metrics in observed.monitor.logs:
        assert "lora_monitor/action/delta_w_norm_mean" in metrics
        if step in [1, 3]:
            assert "train/dmd_loss" in metrics
    if not video_enabled:
        assert observed.monitor.logs[-1][1]["lora_monitor/video/has_lora"] == 0


def test_video_gradients_are_captured_after_sync_before_clipping(tmp_path, monkeypatch):
    from lightx2v_train.trainers.fastwam_action_dmd import trainer as trainer_module

    calls = []

    def synchronized(parameters):
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(0.5)
        calls.append(sum(float(p.grad.double().square().sum()) for p in parameters if p.grad is not None) ** 0.5)

    monkeypatch.setattr(trainer_module, "_all_reduce_gradients", synchronized)
    trainer, expected = _run_tiny_trainer(tmp_path, True)
    assert len(calls) == 3
    for row in _rows(trainer.lora_monitor):
        if row["branch"] == "video" and int(row["step"]) > 0:
            assert float(row["gradient_norm"]) == pytest.approx(expected[(int(row["step"]), "video")])
            assert float(row["gradient_norm"]) == pytest.approx(calls[int(row["step"]) - 1])
