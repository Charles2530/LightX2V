"""Small regression tests for the FasterWAM LightX2V adapter.

These tests deliberately use tiny fake modules; running them must not load the
12GB Wan checkpoint or require a GPU.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from lightx2v_train.model_zoo.native.wan.fasterwam.action_distill import CachedActionDenoiser
from lightx2v_train.trainers.fasterwam_action_consistency.config import FastWAMActionConsistencyConfig
from lightx2v_train.trainers.fasterwam_action_consistency.trainer import FasterWAMActionConsistencyTrainer
from lightx2v_train.utils.registry import build_model, build_trainer


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        return {"tokens": self.proj(action_tokens), "freqs": None, "t_mod": None,
                "context": context, "context_mask": context_mask}

    def post_dit(self, tokens, pre):
        return tokens


class _SparseMoT(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.mixtures = {"action": expert}
        self.seen = None

    def forward_action_with_video_cache(self, **kwargs):
        self.seen = self.mixtures["action"]
        return kwargs["action_tokens"]


class FasterWAMAdapterTest(unittest.TestCase):
    def test_sparse_mot_routes_the_active_expert(self):
        original = _Expert()
        active = _Expert()
        mot = _SparseMoT(original)
        denoiser = CachedActionDenoiser(active, mot)
        condition = SimpleNamespace(context=torch.zeros(1, 1, 2), context_mask=torch.ones(1, 1, dtype=torch.bool),
                                    video_kv_cache=[], attention_mask=torch.ones(1, 1, dtype=torch.bool), video_seq_len=0)
        out = denoiser(torch.zeros(1, 2, 2), torch.ones(1), condition)
        self.assertEqual(tuple(out.shape), (1, 2, 2))
        self.assertIs(mot.seen, active)

    def test_registry_and_config_are_wired(self):
        root = Path(__file__).parents[1]
        config_path = root / "configs/train/fastwam_action_dmd/robotwin_action_1step_consistency_teacher_fasterwam.yaml"
        import yaml
        config = yaml.safe_load(config_path.read_text())
        config.setdefault("logging", {}).setdefault("wandb", {})["enable"] = False
        self.assertEqual(config["model"]["name"], "wan_fasterwam")
        self.assertEqual(config["training"]["method"], "fasterwam_action_consistency")
        parsed = FastWAMActionConsistencyConfig.from_mapping(config)
        self.assertEqual(parsed.flow_target, "teacher")
        self.assertEqual(parsed.supervision_type, "flow")
        self.assertEqual((parsed.teacher_start, parsed.teacher_end), ("t", "0"))
        self.assertEqual(parsed.teacher_reference_steps, 10)
        self.assertAlmostEqual(parsed.flow_loss_weight, 0.2)
        self.assertEqual(type(build_model(config)).__name__, "WanFasterWAMModel")
        self.assertEqual(type(build_trainer(config)).__name__, "FasterWAMActionConsistencyTrainer")

    def test_teacher_target_runs_ten_step_rollout(self):
        class ConstantDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, action, timestep, condition):
                self.calls.append((action.detach().clone(), timestep.detach().clone()))
                return torch.full_like(action, 0.25)

        trainer = FasterWAMActionConsistencyTrainer.__new__(FasterWAMActionConsistencyTrainer)
        trainer.parsed = SimpleNamespace(teacher_reference_steps=10)
        trainer.model = SimpleNamespace(
            unwrap_module=lambda: SimpleNamespace(
                train_action_scheduler=SimpleNamespace(num_train_timesteps=1000)
            )
        )
        trainer.teacher_denoiser = ConstantDenoiser()
        noisy_action = torch.ones(2, 3, 4)
        sigma = torch.tensor([0.5, 0.8])
        first_velocity = trainer.teacher_denoiser(noisy_action, sigma * 1000, None)

        target, x0_target = trainer._teacher_targets(noisy_action, sigma, None, first_velocity)

        self.assertEqual(len(trainer.teacher_denoiser.calls), 10)
        torch.testing.assert_close(target, torch.full_like(target, 0.25))
        torch.testing.assert_close(x0_target, noisy_action - sigma.view(-1, 1, 1) * 0.25)

    def test_checkpoint_shape_filter_keeps_only_compatible_tensors(self):
        from lightx2v_train.model_zoo.wan_fasterwam import WanFasterWAMModel
        model = WanFasterWAMModel.__new__(WanFasterWAMModel)
        target = nn.Linear(2, 2, bias=False)
        model.module = SimpleNamespace(mot=target, unwrap_module=lambda: model.module)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "checkpoint.pt")
            torch.save({"mot": {"weight": torch.ones(2, 2), "bad": torch.ones(3, 3)}}, path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            current = target.state_dict()
            compatible = {k: v for k, v in payload["mot"].items() if k in current and v.shape == current[k].shape}
            target.load_state_dict(compatible, strict=False)
            torch.testing.assert_close(target.weight, torch.ones_like(target.weight))


if __name__ == "__main__":
    unittest.main()
