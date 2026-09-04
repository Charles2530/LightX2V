import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from lightx2v_train.trainers.fastwam_action_consistency.checkpoint import ActionConsistencyCheckpointManager
from lightx2v_train.trainers.fastwam_action_consistency.config import FastWAMActionConsistencyConfig
from lightx2v_train.trainers.fastwam_action_consistency.roles import ActionConsistencyRoles
from lightx2v_train.trainers.fastwam_action_consistency.trainer import _masked_pseudo_huber, shifted_consistency_pair
from torch import nn


class FastWAMActionConsistencyTest(unittest.TestCase):
    def test_requested_configs_parse(self):
        root = Path(__file__).parents[1]
        for name, expected_rank in (
            ("libero_action_1step_consistency.yaml", 64),
            ("robotwin_action_1step_consistency.yaml", 128),
        ):
            with (root / "configs/train/fastwam_action_dmd" / name).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            parsed = FastWAMActionConsistencyConfig.from_mapping(config)
            self.assertEqual(config["training"]["method"], "fastwam_action_consistency")
            self.assertEqual(parsed.target_steps, 2)
            self.assertEqual(parsed.student.lora["rank"], expected_rank)
            self.assertAlmostEqual(parsed.flow_loss_weight, 0.2)

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
                runtime_config={"training": {"method": "fastwam_action_consistency"}},
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
