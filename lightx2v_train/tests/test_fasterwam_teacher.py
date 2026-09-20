import tempfile
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from lightx2v_train.trainers.fasterwam_action_consistency.checkpoint import ActionConsistencyCheckpointManager
from lightx2v_train.trainers.fasterwam_action_consistency.config import FastWAMActionConsistencyConfig
from lightx2v_train.trainers.fasterwam_action_consistency.roles import ActionConsistencyRoles
from lightx2v_train.trainers.fasterwam_action_consistency.trainer import FasterWAMActionConsistencyTrainer, _masked_mse, _masked_pseudo_huber, shifted_consistency_pair
from torch import nn


class _Denoiser(nn.Module):
    def __init__(self, value, dynamic=False):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))
        self.dynamic = dynamic
        self.calls = []

    def forward(self, action, timestep, condition):
        self.calls.append((action.detach().clone(), timestep.detach().clone(), condition, torch.is_grad_enabled()))
        if self.dynamic:
            return self.value * action + (timestep / 1000).view(-1, 1, 1)
        return torch.ones_like(action) * self.value


def _trainer(sigma, source=None, steps=4, supervision=None, **teacher_interval):
    consistency = {"teacher_reference_steps": steps, **teacher_interval}
    if source is not None:
        consistency["flow_target"] = source
    if supervision is not None:
        consistency["supervision_type"] = supervision
    config = {"training": {"student": {"train_type": "full", "optimizer": {}}, "action_consistency": consistency}}
    trainer = FasterWAMActionConsistencyTrainer.__new__(FasterWAMActionConsistencyTrainer)
    trainer.parsed = FastWAMActionConsistencyConfig.from_mapping(config)
    module = SimpleNamespace(train_action_scheduler=SimpleNamespace(num_train_timesteps=1000))
    trainer.model = SimpleNamespace(unwrap_module=lambda: module)
    trainer.student_denoiser = _Denoiser(0.2)
    trainer.teacher_denoiser = _Denoiser(0.3).requires_grad_(False)
    trainer.target_denoiser = _Denoiser(0.5).requires_grad_(False)
    trainer._sigma_pair = lambda action: (sigma, sigma * 0.5)
    return trainer


class FasterWAMTeacherTest(unittest.TestCase):
    def test_requested_configs_parse(self):
        root = Path(__file__).parents[1]
        for name, expected_rank, source, supervision, supervision_weight in (
            ("robotwin_action_1step_consistency_fasterwam.yaml", 128, "data", "flow", 0.0),
            ("robotwin_action_1step_consistency_teacher_fasterwam.yaml", 128, "teacher", "flow", 0.2),
            ("libero_action_1step_consistency_fasterwam.yaml", 128, "data", "flow", 0.2),
            ("libero_action_1step_consistency_fasterwam_teacher.yaml", 128, "teacher", "flow", 0.2),
        ):
            with (root / "configs/train/fastwam_action_dmd" / name).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            parsed = FastWAMActionConsistencyConfig.from_mapping(config)
            self.assertEqual(config["training"]["method"], "fasterwam_action_consistency")
            self.assertEqual(parsed.target_steps, 10)
            self.assertEqual(parsed.teacher_reference_steps, 10)
            self.assertEqual(parsed.flow_target, source)
            self.assertEqual(parsed.supervision_type, supervision)
            self.assertEqual((parsed.teacher_start, parsed.teacher_end), ("t", "0"))
            self.assertEqual(parsed.student.lora["rank"], expected_rank)
            self.assertAlmostEqual(parsed.flow_loss_weight, supervision_weight)
            if source == "teacher":
                self.assertTrue(config["training"]["output_dir"].endswith("_teacher_future"))
                self.assertTrue(config["logging"]["wandb"]["name"])
            if name.startswith("libero_"):
                self.assertEqual(config["model"]["proprio_dim"], 8)
                self.assertEqual(config["model"]["action_dit_config"]["action_dim"], 7)
                self.assertFalse(config["model"]["mot_checkpoint_mixed_attn"])
                self.assertFalse(config["data"]["train"]["observation_only_video"])

    def test_teacher_interval_validation(self):
        for start, end in (("t", "0"), ("1", "0"), ("t", "r"), (1, 0)):
            parsed = _trainer(torch.tensor([0.5]), teacher_start=start, teacher_end=end).parsed
            self.assertEqual((parsed.teacher_start, parsed.teacher_end), (str(start), str(end)))
        for start, end in (("1", "r"), ("0", "0"), ("t", "t"), ("T", "0"), ("t", "1"), (None, "0")):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "teacher_start/teacher_end"):
                _trainer(torch.tensor([0.5]), teacher_start=start, teacher_end=end)

    def test_invalid_flow_target_is_rejected(self):
        for source in ("ema", "DATA", "", 1):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "flow_target"):
                _trainer(torch.tensor([0.5]), source)

    def test_invalid_supervision_type_is_rejected(self):
        for supervision in ("velocity", "X0", "", 1):
            with self.subTest(supervision=supervision), self.assertRaisesRegex(ValueError, "supervision_type"):
                _trainer(torch.tensor([0.5]), supervision=supervision)

    def test_training_details_include_supervision_type(self):
        for supervision, (start, end) in product(("flow", "x0"), (("t", "0"), ("1", "0"), ("t", "r"))):
            trainer = _trainer(torch.tensor([0.5]), "teacher", supervision=supervision, teacher_start=start, teacher_end=end)
            self.assertIn(f"supervision_type={supervision}", trainer._training_details())
            self.assertIn(f"teacher_start={start} teacher_end={end}", trainer._training_details())

    def test_default_and_flow_preserve_loss_gradient_and_rng(self):
        action = torch.tensor([[[0.1], [0.4]], [[-0.2], [0.8]]])
        sigma = torch.tensor([0.8, 0.4])
        valid = torch.tensor([[True, False], [True, True]])
        for source, supervision, interval in product((None, "data", "teacher"), (None, "flow"), ({}, {"teacher_start": "t", "teacher_end": "0"})):
            with self.subTest(source=source, supervision=supervision, interval=interval):
                trainer = _trainer(sigma, source, supervision=supervision, **interval)
                torch.manual_seed(9)
                noise = torch.randn_like(action)
                expected_rng = torch.get_rng_state()
                value = torch.tensor(0.2, requires_grad=True)
                # Constant teacher=0.3, EMA=0.5: f_student - f_EMA has a closed form.
                difference = sigma.view(-1, 1, 1) * (0.3 - value) + (sigma * 0.5).view(-1, 1, 1) * 0.2
                expected_consistency = _masked_pseudo_huber(difference.expand_as(action), torch.zeros_like(action), valid, 0.001)
                flow_target = torch.full_like(action, 0.3) if source == "teacher" else noise - action
                expected_flow = _masked_mse(value.expand_as(action), flow_target, valid)
                expected = expected_consistency + 0.2 * expected_flow
                expected.backward()
                torch.manual_seed(9)
                loss, metrics = trainer._loss({"action": action}, None, valid)
                loss.backward()
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(metrics["consistency"], expected_consistency)
                torch.testing.assert_close(metrics["flow"], expected_flow)
                torch.testing.assert_close(trainer.student_denoiser.value.grad, value.grad)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                self.assertEqual(len(trainer.teacher_denoiser.calls), 4 if source == "teacher" else 1)

    def test_teacher_rollout_inputs_select_state_time_and_reuse_noise(self):
        for dtype, (start, end) in product((torch.float32, torch.bfloat16), (("t", "0"), ("1", "0"), ("t", "r"))):
            with self.subTest(dtype=dtype, start=start, end=end):
                sigma = torch.tensor([0.25, 0.8], dtype=dtype)
                noise = torch.tensor([[[1.0]], [[-0.5]]], dtype=dtype)
                noisy = torch.tensor([[[0.5]], [[0.25]]], dtype=dtype)
                trainer = _trainer(sigma, "teacher", teacher_start=start, teacher_end=end)
                trainer.teacher_denoiser.dynamic = True
                condition = object()
                with torch.no_grad():
                    first = trainer.teacher_denoiser(noisy, sigma * 1000, condition)
                first_before = first.clone()
                torch.manual_seed(31)
                expected_end = torch.rand(sigma.shape) * sigma.float() if end == "r" else torch.zeros_like(sigma, dtype=torch.float32)
                expected_rng = torch.get_rng_state()
                torch.manual_seed(31)
                state, rollout_start, rollout_end, velocity = trainer._teacher_rollout_inputs(noise, noisy, sigma, condition, first)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                torch.testing.assert_close(rollout_end, expected_end, rtol=0, atol=0)
                self.assertEqual(rollout_end.dtype, torch.float32)
                self.assertTrue(torch.all((rollout_end >= 0) & (rollout_end <= rollout_start)))
                torch.testing.assert_close(first, first_before, rtol=0, atol=0)
                if start == "1":
                    self.assertIs(state, noise)
                    torch.testing.assert_close(rollout_start, torch.ones_like(sigma))
                    self.assertEqual(len(trainer.teacher_denoiser.calls), 2)
                    seen_action, seen_t, seen_condition, grad_enabled = trainer.teacher_denoiser.calls[-1]
                    torch.testing.assert_close(seen_action, noise)
                    torch.testing.assert_close(seen_t, torch.full_like(sigma, 1000))
                    self.assertIs(seen_condition, condition)
                    self.assertFalse(grad_enabled)
                    torch.testing.assert_close(velocity, 0.3 * noise + 1)
                else:
                    self.assertIs(state, noisy)
                    self.assertIs(rollout_start, sigma)
                    self.assertIs(velocity, first)
                    self.assertEqual(len(trainer.teacher_denoiser.calls), 1)

    def test_teacher_rollout_matches_euler_endpoint_and_reuses_first_call(self):
        for dtype, steps in product((torch.float32, torch.bfloat16), (1, 4, 10)):
            for interval in ("t0", "10", "tr"):
                with self.subTest(dtype=dtype, steps=steps, interval=interval):
                    sigma = torch.ones(2, dtype=dtype) if interval == "10" else torch.tensor([0.25, 0.8], dtype=dtype)
                    end = torch.tensor([0.125, 0.375]) if interval == "tr" else torch.zeros(2)
                    trainer = _trainer(sigma, "teacher", steps)
                    trainer.teacher_denoiser.dynamic = True
                    trainer.teacher_denoiser.value.fill_(0.25)  # Exactly representable in BF16 and FP32.
                    noisy = torch.tensor([[[1.0, -0.5]], [[0.2, 0.7]]], dtype=dtype)
                    original = noisy.clone()
                    condition = object()
                    with torch.no_grad():
                        first = trainer.teacher_denoiser(noisy, sigma * 1000, condition)
                    first_before = first.clone()
                    kwargs = {} if interval == "t0" else {"sigma_end": end}
                    target, x0_target = trainer._teacher_targets(noisy, sigma, condition, first, **kwargs)
                    self.assertEqual(len(trainer.teacher_denoiser.calls), steps)
                    self.assertEqual(target.dtype, torch.float32)
                    self.assertFalse(target.requires_grad)
                    self.assertEqual(x0_target.dtype, torch.float32)
                    self.assertFalse(x0_target.requires_grad)
                    torch.testing.assert_close(noisy, original)
                    torch.testing.assert_close(first, first_before)

                    # Independently integrate the field; use endpoint displacement as the oracle.
                    state = noisy.float()
                    for index, (seen_action, seen_t, seen_condition, grad_enabled) in enumerate(trainer.teacher_denoiser.calls):
                        timestep = sigma * 1000 if index == 0 else ((sigma.float() * (1 - index / steps) + end * (index / steps)) * 1000).to(dtype)
                        torch.testing.assert_close(seen_action, state.to(dtype))
                        torch.testing.assert_close(seen_t, timestep)
                        self.assertIs(seen_condition, condition)
                        self.assertFalse(grad_enabled)
                        velocity = 0.25 * state.to(dtype) + (timestep / 1000).view(-1, 1, 1)
                        state = state - ((sigma.float() - end) / steps).view(-1, 1, 1) * velocity.float()
                    expected = (noisy.float() - state) / (sigma.float() - end).view(-1, 1, 1)
                    torch.testing.assert_close(target, expected)
                    torch.testing.assert_close(x0_target, state - end.view(-1, 1, 1) * expected)
                    if interval != "tr":
                        torch.testing.assert_close(x0_target, state, rtol=0, atol=0)
                    if steps == 1:
                        torch.testing.assert_close(target, first.float())

    def test_teacher_target_is_stable_at_zero_and_tiny_sigma(self):
        for dtype, end_ratio in product((torch.float32, torch.bfloat16), (0.0, 0.999999, 1.0)):
            with self.subTest(dtype=dtype, end_ratio=end_ratio):
                sigma = torch.tensor([0.0, 1e-12, 1e-6, 0.75], dtype=dtype)
                trainer = _trainer(sigma, "teacher", steps=10)
                noisy = torch.ones(4, 2, 2, dtype=dtype)
                with torch.no_grad():
                    first = trainer.teacher_denoiser(noisy, sigma * 1000, None)
                target, x0_target = trainer._teacher_targets(noisy, sigma, None, first, sigma_end=sigma.float() * end_ratio)
                self.assertTrue(torch.isfinite(target).all())
                self.assertTrue(torch.isfinite(x0_target).all())
                torch.testing.assert_close(target, first.float())
                torch.testing.assert_close(x0_target[0], noisy[0].float())
                torch.testing.assert_close(x0_target, noisy.float() - sigma.float().view(-1, 1, 1) * first.float())

    def test_teacher_loss_mask_and_student_only_gradients(self):
        action = torch.tensor([[[0.1], [0.4]], [[-0.2], [0.8]]])
        sigma = torch.tensor([0.3, 0.9])
        valid = torch.tensor([[True, False], [True, True]])
        results = []
        for offset in (0.0, 100.0):
            trainer = _trainer(sigma, "teacher", steps=4)
            trainer.teacher_denoiser.dynamic = True
            condition = object()
            sample = action + (~valid).unsqueeze(-1) * offset
            torch.manual_seed(3)
            noise = torch.randn_like(action)
            expected_rng = torch.get_rng_state()
            torch.manual_seed(3)
            loss, metrics = trainer._loss({"action": sample}, condition, valid)
            loss.backward()
            results.append((loss.detach(), trainer.student_denoiser.value.grad.clone()))
            torch.testing.assert_close(torch.get_rng_state(), expected_rng)
            self.assertEqual(set(metrics), {"consistency", "flow", "x0"})
            calls = trainer.teacher_denoiser.calls
            self.assertEqual(len(calls), 4)
            torch.testing.assert_close(calls[0][0], (1 - sigma.view(-1, 1, 1)) * sample + sigma.view(-1, 1, 1) * noise)
            self.assertTrue(all(call[2] is condition and not call[3] for call in calls))
            self.assertEqual(len(trainer.student_denoiser.calls), 1)
            self.assertIsNone(trainer.teacher_denoiser.value.grad)
            self.assertIsNone(trainer.target_denoiser.value.grad)
            expected_target = torch.stack([0.3 * call[0] + (call[1] / 1000).view(-1, 1, 1) for call in calls]).mean(0)
            expected_flow = _masked_mse(torch.full_like(action, 0.2), expected_target, valid)
            torch.testing.assert_close(metrics["flow"], expected_flow)
            self.assertGreater(trainer.student_denoiser.value.grad.abs().item(), 0)
        torch.testing.assert_close(results[0], results[1])

    def test_teacher_intervals_preserve_student_consistency_and_loss_masks(self):
        for dtype, (start, end), mask_kind in product((torch.float32, torch.bfloat16), (("t", "0"), ("1", "0"), ("t", "r")), ("none", "partial", "padding")):
            with self.subTest(dtype=dtype, start=start, end=end, mask=mask_kind):
                action = torch.tensor([[[0.125], [0.5]], [[-0.25], [0.75]]], dtype=dtype)
                sigma = torch.tensor([0.25, 0.75], dtype=dtype)
                valid = None if mask_kind == "none" else torch.tensor([[True, False], [True, True]])
                if mask_kind == "padding":
                    valid.zero_()
                baseline = _trainer(sigma, "teacher")
                baseline.teacher_denoiser.dynamic = True
                baseline.teacher_denoiser.value.fill_(0.25)  # Exactly representable in BF16.
                torch.manual_seed(17)
                _, baseline_metrics = baseline._loss({"action": action}, None, valid)

                trainer = _trainer(sigma, "teacher", teacher_start=start, teacher_end=end)
                trainer.teacher_denoiser.dynamic = True
                trainer.teacher_denoiser.value.fill_(0.25)
                torch.manual_seed(17)
                torch.randn_like(action)
                rollout_end = torch.rand(sigma.shape) * sigma.float() if end == "r" else torch.zeros_like(sigma, dtype=torch.float32)
                expected_rng = torch.get_rng_state()
                torch.manual_seed(17)
                loss, metrics = trainer._loss({"action": action}, None, valid)
                loss.backward()

                torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                torch.testing.assert_close(metrics["consistency"], baseline_metrics["consistency"], rtol=0, atol=0)
                for role in ("student_denoiser", "target_denoiser"):
                    calls = getattr(trainer, role).calls
                    self.assertEqual(len(calls), 1)
                    torch.testing.assert_close(calls[0][:2], getattr(baseline, role).calls[0][:2], rtol=0, atol=0)

                calls = trainer.teacher_denoiser.calls
                self.assertEqual(len(calls), 5 if start == "1" else 4)
                self.assertTrue(all(not call[3] for call in calls))
                rollout_calls = calls[1:] if start == "1" else calls
                velocities = [(0.25 * call[0] + (call[1] / 1000).view(-1, 1, 1)).float() for call in rollout_calls]
                mean_velocity = torch.stack(velocities).mean(0)
                rollout_start = torch.ones_like(sigma).float() if start == "1" else sigma.float()
                state = rollout_calls[0][0].float()
                for velocity in velocities:
                    state = state - ((rollout_start - rollout_end) / 4).view(-1, 1, 1) * velocity
                x0_target = state - rollout_end.view(-1, 1, 1) * mean_velocity
                student_velocity = torch.full_like(action, 0.2)
                student_x0 = trainer.student_denoiser.calls[0][0] - sigma.view(-1, 1, 1) * student_velocity
                torch.testing.assert_close(metrics["flow"], _masked_mse(student_velocity, mean_velocity, valid))
                torch.testing.assert_close(metrics["x0"], _masked_mse(student_x0, x0_target, valid))
                torch.testing.assert_close(loss, metrics["consistency"] + 0.2 * metrics["flow"])
                self.assertEqual(set(metrics), {"consistency", "flow", "x0"})
                self.assertTrue(all(not metric.requires_grad for metric in metrics.values()))
                self.assertIsNone(trainer.teacher_denoiser.value.grad)
                self.assertIsNone(trainer.target_denoiser.value.grad)
                self.assertTrue(torch.isfinite(trainer.student_denoiser.value.grad))
                if mask_kind == "padding":
                    self.assertEqual(loss.item(), 0.0)
                    self.assertEqual(trainer.student_denoiser.value.grad.item(), 0.0)

    def test_data_targets_ignore_teacher_interval(self):
        action = torch.ones(2, 2, 1)
        results = []
        for start, end in (("t", "0"), ("1", "0"), ("t", "r")):
            trainer = _trainer(torch.tensor([0.25, 0.75]), "data", teacher_start=start, teacher_end=end)
            torch.manual_seed(17)
            loss, metrics = trainer._loss({"action": action}, None, None)
            loss.backward()
            results.append((loss.detach(), metrics, trainer.student_denoiser.value.grad, torch.get_rng_state()))
            self.assertEqual(len(trainer.teacher_denoiser.calls), 1)
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)

    def test_x0_loss_uses_selected_target_weight_mask_and_student_only_gradients(self):
        for source, dtype, mask_kind in product(("data", "teacher"), (torch.float32, torch.bfloat16), ("none", "partial", "padding")):
            with self.subTest(source=source, dtype=dtype, mask=mask_kind):
                action = torch.tensor([[[0.125], [0.5]], [[-0.25], [0.75]]], dtype=dtype)
                sigma = torch.tensor([0.75, 0.25], dtype=dtype)
                expanded = sigma.view(-1, 1, 1)
                valid = None
                if mask_kind == "partial":
                    valid = torch.tensor([[True, False], [True, True]])
                elif mask_kind == "padding":
                    valid = torch.zeros(2, 2, dtype=torch.bool)
                trainer = _trainer(sigma, source, steps=4, supervision="x0")
                trainer.parsed = replace(trainer.parsed, consistency_loss_weight=1.7, flow_loss_weight=0.6)
                trainer.teacher_denoiser.dynamic = True
                trainer.teacher_denoiser.value.fill_(0.25)
                torch.manual_seed(17)
                noise = torch.randn_like(action)
                expected_rng = torch.get_rng_state()
                noisy = (1 - expanded) * action + expanded * noise

                # Integrate the known teacher field independently in FP32.
                state = noisy.float()
                velocities = []
                for index in range(4):
                    timestep = sigma * 1000 if index == 0 else (sigma.float() * (1 - index / 4) * 1000).to(dtype)
                    velocity = 0.25 * state.to(dtype) + (timestep / 1000).view(-1, 1, 1)
                    velocities.append(velocity.float())
                    state = state - (expanded.float() / 4) * velocity.float()
                x0_target = state if source == "teacher" else action
                flow_target = torch.stack(velocities).mean(0) if source == "teacher" else noise - action

                value = torch.tensor(0.2, requires_grad=True)
                student_velocity = torch.ones_like(action) * value
                student_x0 = noisy - expanded * student_velocity
                endpoint = noisy + (expanded * 0.5 - expanded) * velocities[0].to(dtype)
                ema_x0 = endpoint - expanded * 0.5 * torch.full_like(action, 0.5)
                expected_x0 = _masked_mse(student_x0, x0_target, valid)
                expected_consistency = _masked_pseudo_huber(student_x0, ema_x0, valid, 0.001)
                expected = 1.7 * expected_consistency + 0.6 * expected_x0
                expected.backward()

                torch.manual_seed(17)
                loss, metrics = trainer._loss({"action": action}, None, valid)
                loss.backward()
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(metrics["x0"], expected_x0)
                torch.testing.assert_close(metrics["consistency"], expected_consistency)
                torch.testing.assert_close(metrics["flow"], _masked_mse(student_velocity, flow_target, valid))
                torch.testing.assert_close(trainer.student_denoiser.value.grad, value.grad)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                self.assertEqual(set(metrics), {"consistency", "flow", "x0"})
                self.assertTrue(all(not metric.requires_grad for metric in metrics.values()))
                self.assertEqual(len(trainer.teacher_denoiser.calls), 4 if source == "teacher" else 1)
                self.assertTrue(all(not call[3] for call in trainer.teacher_denoiser.calls))
                self.assertEqual(len(trainer.student_denoiser.calls), 1)
                self.assertEqual(len(trainer.target_denoiser.calls), 1)
                self.assertIsNone(trainer.teacher_denoiser.value.grad)
                self.assertIsNone(trainer.target_denoiser.value.grad)
                if mask_kind == "padding":
                    self.assertEqual(loss.item(), 0.0)
                    self.assertEqual(trainer.student_denoiser.value.grad.item(), 0.0)
                    self.assertTrue(all(metric.item() == 0.0 for metric in metrics.values()))

    def test_x0_supervision_is_stable_at_zero_and_tiny_sigma(self):
        for source, dtype in product(("data", "teacher"), (torch.float32, torch.bfloat16)):
            with self.subTest(source=source, dtype=dtype):
                sigma = torch.tensor([0.0, 1e-12, 1e-6], dtype=dtype)
                trainer = _trainer(sigma, source, steps=10, supervision="x0")
                loss, metrics = trainer._loss({"action": torch.ones(3, 2, 2, dtype=dtype)}, None, None)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(metric) for metric in metrics.values()))
                self.assertTrue(torch.isfinite(trainer.student_denoiser.value.grad))

    def test_shifted_pair_uses_two_step_stride(self):
        base = torch.tensor([1.0, 0.75, 0.5, 0.25])
        start, end = shifted_consistency_pair(base, shift=5.0, target_steps=2)
        expected_end = 5.0 * torch.tensor([0.5, 0.25, 0.0, 0.0]) / (1.0 + 4.0 * torch.tensor([0.5, 0.25, 0.0, 0.0]))
        torch.testing.assert_close(end, expected_end)
        self.assertTrue(torch.all(end <= start))

    def test_x0_consistency_identity_and_masked_loss(self):
        action = torch.randn(2, 3, 4)
        noise = torch.randn_like(action)
        sigma = torch.tensor([0.2, 0.9]).view(2, 1, 1)
        noisy = (1.0 - sigma) * action + sigma * noise
        predicted_x0 = noisy - sigma * (noise - action)
        torch.testing.assert_close(predicted_x0, action)

        changed_only_under_mask = action.clone()
        changed_only_under_mask[:, 1:] += 10.0
        valid = torch.tensor([[True, False, False], [True, False, False]])
        self.assertEqual(float(_masked_pseudo_huber(action, changed_only_under_mask, valid, 0.001)), 0.0)

    def test_ema_updates_without_changing_teacher(self):
        config = SimpleNamespace(train_type="full", lora=None)
        expert = nn.Linear(2, 2, bias=False)
        roles = ActionConsistencyRoles.build(expert, config)
        teacher_before = roles.teacher.weight.detach().clone()
        target_before = roles.target.weight.detach().clone()
        with torch.no_grad():
            roles.student.weight.add_(2.0)
        roles.update_target(0.5)
        torch.testing.assert_close(roles.target.weight, target_before + 1.0)
        torch.testing.assert_close(roles.teacher.weight, teacher_before)

    def test_checkpoint_round_trip_restores_online_and_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(train_type="full", lora=None)
            roles = ActionConsistencyRoles.build(nn.Linear(2, 2, bias=False), config)
            optimizer = torch.optim.AdamW(roles.student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
            trainer = SimpleNamespace(
                config={"resume": {}},
                runtime_config={"training": {"method": "fasterwam_action_consistency"}},
                output_dir=directory,
                save_total_limit=2,
                parsed=SimpleNamespace(student=config),
                roles=roles,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            manager = ActionConsistencyCheckpointManager(trainer)
            student_before = roles.student.weight.detach().clone()
            target_before = roles.target.weight.detach().clone()
            manager.save(7)
            with torch.no_grad():
                roles.student.weight.zero_()
                roles.target.weight.zero_()
            self.assertEqual(manager.load(str(Path(directory) / "checkpoint-000000007")), 7)
            torch.testing.assert_close(roles.student.weight, student_before)
            torch.testing.assert_close(roles.target.weight, target_before)


if __name__ == "__main__":
    unittest.main()
