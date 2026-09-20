"""CPU regressions for the inference-aligned FasterWAM conditioning path."""
from types import SimpleNamespace
from pathlib import Path
import unittest
import tempfile
import importlib.util
from unittest.mock import Mock

import torch
import yaml

from lightx2v_train.model_zoo.native.wan.fasterwam.action_distill import build_action_distill_condition
from lightx2v_train.trainers.fasterwam_action_consistency.config import FastWAMActionConsistencyConfig
from lightx2v_train.trainers.fasterwam_action_consistency.checkpoint import ActionConsistencyCheckpointManager
from lightx2v_train.trainers.fasterwam_action_consistency.trainer import FasterWAMActionConsistencyTrainer


class FutureCacheTest(unittest.TestCase):
    def setUp(self):
        self.seen = []
        def pre_dit(**kwargs):
            self.seen.append(kwargs)
            x = kwargs['x']
            return dict(tokens=x.flatten(2).transpose(1, 2), freqs=None, t_mod=None,
                        context=kwargs['context'], context_mask=kwargs['context_mask'], meta={'tokens_per_frame': 4})
        self.model = SimpleNamespace(
            video_expert=SimpleNamespace(pre_dit=pre_dit, fuse_vae_embedding_in_latents=True,
                                         build_video_to_video_mask=Mock(return_value=torch.ones(4, 4))),
            infer_video_scheduler=SimpleNamespace(build_inference_schedule=lambda **kw: (torch.tensor([1000.]), None)),
            mot=SimpleNamespace(prefill_video_cache=lambda **kw: [{'k': kw['video_tokens']}, None]),
            _build_video_attention_mask=Mock(return_value=torch.ones(12, 12)),
            _build_mot_attention_masks=Mock(return_value=[torch.ones(14, 14), torch.eye(14)]),
            _build_mot_attention_mask=Mock(return_value=torch.ones(6, 6)),
        )
        self.inputs = dict(first_frame_latents=torch.full((1, 2, 1, 2, 2), 7.),
                           input_latents=torch.full((1, 2, 3, 2, 2), 999.),
                           context=torch.zeros(1, 2, 3), context_mask=torch.ones(1, 2), action=torch.zeros(1, 2, 2))

    def build(self, seed=42):
        return build_action_distill_condition(self.model, self.inputs, video_conditioning='one_pass_future_cache',
                                              generator=torch.Generator().manual_seed(seed))

    def test_future_noise_timestep_sparse_masks_and_no_gt_leak(self):
        condition = self.build()
        self.assertEqual(condition.video_seq_len, 12)
        self.assertEqual(self.seen[-1]['timestep'].item(), 1000.)
        torch.testing.assert_close(self.seen[-1]['x'][:, :, :1], self.inputs['first_frame_latents'])
        expected = torch.randn((1, 2, 3, 2, 2), generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(self.seen[-1]['x'][:, :, 1:], expected[:, :, 1:])
        self.inputs['input_latents'].fill_(-1000.)
        repeated = self.build()
        torch.testing.assert_close(condition.video_kv_cache[0]['k'], repeated.video_kv_cache[0]['k'])
        other = self.build(seed=43)
        self.assertFalse(torch.equal(condition.video_kv_cache[0]['k'], other.video_kv_cache[0]['k']))
        self.model._build_mot_attention_masks.assert_called()
        self.model._build_mot_attention_mask.assert_not_called()
        detached = condition.detach()
        self.assertIsNone(detached.video_kv_cache[1])
        self.assertIsInstance(detached.attention_mask, list)

    def test_legacy_default_remains_observation_only(self):
        condition = build_action_distill_condition(self.model, self.inputs)
        self.assertEqual(condition.video_seq_len, 4)
        self.assertEqual(self.seen[-1]['timestep'].item(), 0.)
        self.assertIsInstance(condition.detach().attention_mask, torch.Tensor)

    def test_no_future_frames_fails_fast(self):
        self.inputs['input_latents'] = self.inputs['first_frame_latents']
        with self.assertRaisesRegex(ValueError, 'multi-frame'):
            self.build()

    def test_recipe_and_legacy_config(self):
        cfg = yaml.safe_load((Path(__file__).parents[1] / 'configs/train/fastwam_action_dmd/robotwin_action_1step_consistency_teacher_fasterwam.yaml').read_text())
        parsed = FastWAMActionConsistencyConfig.from_mapping(cfg)
        self.assertEqual(parsed.video_conditioning, 'one_pass_future_cache')
        self.assertEqual((parsed.teacher_start, parsed.teacher_end), ('t', '0'))
        for split in ('train', 'val'):
            self.assertEqual(cfg['data'][split]['delta_action_dim_mask'], [False] * 14)
        cfg['training']['action_consistency'].pop('video_conditioning')
        self.assertEqual(FastWAMActionConsistencyConfig.from_mapping(cfg).video_conditioning, 'observation_only')
        cfg['training']['action_consistency']['video_conditioning'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'video_conditioning'):
            FastWAMActionConsistencyConfig.from_mapping(cfg)

    def test_plain_consistency_recipe_uses_future_cache_without_auxiliary_loss(self):
        cfg = yaml.safe_load((Path(__file__).parents[1] / 'configs/train/fastwam_action_dmd/robotwin_action_1step_consistency_fasterwam.yaml').read_text())
        parsed = FastWAMActionConsistencyConfig.from_mapping(cfg)
        self.assertEqual(parsed.video_conditioning, 'one_pass_future_cache')
        self.assertEqual(parsed.consistency_loss_weight, 1.0)
        self.assertEqual(parsed.flow_loss_weight, 0.0)
        self.assertEqual(parsed.flow_target, 'data')
        self.assertEqual(parsed.target_steps, 10)
        for split in ('train', 'val'):
            self.assertEqual(cfg['data'][split]['delta_action_dim_mask'], [False] * 14)
        self.assertEqual(cfg['training']['output_dir'], 'runs/fasterwam_robotwin_action_1step_consistency_ts10_future')

    def test_libero_recipes_use_future_cache_and_preserve_objectives(self):
        from lightx2v_train.data.libero.processor import FastWAMProcessor
        root = Path(__file__).parents[1] / 'configs/train/fastwam_action_dmd'
        for name, target, weight in (
            ('libero_action_1step_consistency_fasterwam.yaml', 'data', 0.0),
            ('libero_action_1step_consistency_fasterwam_teacher.yaml', 'teacher', 0.2),
        ):
            with self.subTest(config=name):
                cfg = yaml.safe_load((root / name).read_text())
                parsed = FastWAMActionConsistencyConfig.from_mapping(cfg)
                self.assertEqual(parsed.video_conditioning, 'one_pass_future_cache')
                self.assertEqual(parsed.flow_target, target)
                self.assertEqual(parsed.flow_loss_weight, weight)
                self.assertEqual(parsed.consistency_loss_weight, 1.0)
                for split in ('train', 'val'):
                    data = cfg['data'][split]
                    processor = FastWAMProcessor(data['shape_meta'], data['num_frames'],
                        delta_action_dim_mask=data.get('delta_action_dim_mask', (True,) * 6 + (False,)))
                    self.assertEqual(processor.delta_action_dim_mask.tolist(), [True] * 6 + [False])

    def test_old_checkpoint_cannot_silently_resume_with_new_conditioning(self):
        trainer = SimpleNamespace(parsed=SimpleNamespace(video_conditioning='one_pass_future_cache'))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.yaml').write_text('training: {}\n')
            with self.assertRaisesRegex(RuntimeError, 'Checkpoint video_conditioning=observation_only'):
                ActionConsistencyCheckpointManager(trainer).load(directory)

    def test_trainer_prefill_uses_native_precision_even_inside_autocast(self):
        self.inputs['action_is_pad'] = None
        def build_inputs(sample):
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            return self.inputs
        self.model.build_inputs = build_inputs
        trainer = FasterWAMActionConsistencyTrainer.__new__(FasterWAMActionConsistencyTrainer)
        trainer.parsed = SimpleNamespace(video_conditioning='one_pass_future_cache')
        trainer.model = SimpleNamespace(unwrap_module=lambda: self.model, device=torch.device('cpu'))
        before = torch.get_rng_state().clone()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, condition, _ = trainer._prepare_batch({}, video_generator=torch.Generator().manual_seed(42))
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(condition.video_seq_len, 12)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_fasterwam_caches_are_separate_from_fastwam_for_both_datasets(self):
        root = Path(__file__).parents[1] / 'configs/train/fastwam_action_dmd'
        for name, dataset in (
            ('robotwin_action_1step_consistency_fasterwam.yaml', 'robotwin'),
            ('robotwin_action_1step_consistency_teacher_fasterwam.yaml', 'robotwin'),
            ('libero_action_1step_consistency_fasterwam.yaml', 'libero'),
            ('libero_action_1step_consistency_fasterwam_teacher.yaml', 'libero'),
        ):
            cfg = yaml.safe_load((root / name).read_text())
            teacher_suffix = '_teacher' if 'teacher' in name else ''
            self.assertEqual(cfg['training']['output_dir'], f'runs/fasterwam_{dataset}_action_1step_consistency_ts10{teacher_suffix}_future')
            self.assertTrue(cfg['logging']['wandb']['name'].endswith('-future'))
            for split in ('train', 'val'):
                cache = cfg['data'][split]['text_embedding_cache_dir']
                self.assertTrue(cache.endswith(f'/runs/fasterwam_{dataset}_cache/text_embeds_cache'), (name, split, cache))

    def test_cache_generator_honors_config_and_rejects_stale_write_override(self):
        script = Path(__file__).parents[1] / 'tools/precompute_robotwin_text_cache.py'
        spec = importlib.util.spec_from_file_location('cache_preflight', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = dict(model={'name': 'wan_fasterwam'}, training={'output_dir': '/tmp/experiment'},
                   data={'train': {'text_embedding_cache_dir': '/tmp/fasterwam_separate_cache'}})
        self.assertEqual(module.resolve_cache_dir(cfg, environment={}), Path('/tmp/fasterwam_separate_cache'))
        env = {'FASTWAM_TEXT_CACHE_DIR': '/tmp/fastwam_original_cache'}
        with self.assertRaisesRegex(ValueError, 'original FastWAM cache'):
            module.resolve_cache_dir(cfg, environment=env)
        self.assertEqual(module.resolve_cache_dir(cfg, environment=env, validate_only=True), Path(env['FASTWAM_TEXT_CACHE_DIR']))
        cfg['model']['name'] = 'wan_fastwam'
        self.assertEqual(module.resolve_cache_dir(cfg, environment={}), Path('/tmp/experiment/text_embeds_cache'))


if __name__ == '__main__':
    unittest.main()
