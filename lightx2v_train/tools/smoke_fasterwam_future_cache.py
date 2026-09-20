"""Real-data future-KV parity, DDP training, validation and checkpoint smoke.

Run with torchrun --standalone --nproc_per_node=8. Outputs go to a NEW directory;
W&B is disabled and existing training/evaluation runs are never resumed.
"""
import argparse
import faulthandler
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
import yaml

from lightx2v_train.data import build_data, prepare_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.model_zoo.native.wan.fasterwam.action_distill import (
    CachedActionDenoiser, sample_action_teacher,
)
from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config
from lightx2v_train.trainers import build_trainer


def digest(parameters):
    result = hashlib.sha256()
    for parameter in parameters:
        result.update(parameter.detach().float().cpu().numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def check_parity(adapter, sample, trainer):
    model = adapter.unwrap_module().eval().requires_grad_(False)
    # Exercise the actual trainer precision context, not just the standalone builder.
    inputs, condition, _ = trainer._prepare_batch(sample, video_generator=torch.Generator().manual_seed(42))
    captured = {}
    original_predict = model._predict_action_noise_with_cache
    def capture(self, **kwargs):
        if not captured:
            captured.update(kwargs)
        return original_predict(**kwargs)
    model._predict_action_noise_with_cache = MethodType(capture, model)
    try:
        upstream = model.infer_action_one_pass_future_cache(
            prompt=None, input_image=sample['video'][:, :, 0],
            action_horizon=inputs['action'].shape[1], num_video_frames=sample['video'].shape[2],
            proprio=sample['proprio'][:, 0], context=sample['context'], context_mask=sample['context_mask'],
            seed=42, num_inference_steps=10,
        )['action']
    finally:
        model._predict_action_noise_with_cache = original_predict
    # Token count depends on the dataset resolution; 360 is RoboTwin-specific.
    latent_shape = inputs['input_latents'].shape[2:]
    patch_size = model.video_expert.patch_size
    assert len(latent_shape) == len(patch_size) == 3
    assert all(size % patch == 0 for size, patch in zip(latent_shape, patch_size))
    expected_tokens = math.prod(size // patch for size, patch in zip(latent_shape, patch_size))
    assert latent_shape[0] > 1
    assert condition.video_seq_len == captured['video_seq_len'] == expected_tokens
    cache_max_abs = 0.0
    assert len(condition.video_kv_cache) == len(captured['video_kv_cache'])
    for actual, expected in zip(condition.video_kv_cache, captured['video_kv_cache']):
        if expected is None:
            assert actual is None
            continue
        assert set(actual) == set(expected)
        for key in actual:
            cache_max_abs = max(cache_max_abs, float((actual[key] - expected[key]).abs().max()))
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert len(condition.attention_mask) == len(captured['attention_mask'])
    for actual, expected in zip(condition.attention_mask, captured['attention_mask']):
        assert torch.equal(actual, expected)
    noise = torch.randn(inputs['action'].shape, generator=torch.Generator().manual_seed(42)).to(inputs['action'])
    denoiser = CachedActionDenoiser(model.action_expert, model.mot)
    reproduced = sample_action_teacher(denoiser, noise, condition, model.infer_action_scheduler, 10)[0].float().cpu()
    torch.testing.assert_close(reproduced, upstream, rtol=0, atol=0)
    detached = condition.detach()
    assert not any(value.requires_grad for layer in detached.video_kv_cache if layer for value in layer.values())
    return dict(video_tokens=condition.video_seq_len,
                cached_layers=sum(layer is not None for layer in condition.video_kv_cache),
                cache_max_abs=cache_max_abs, teacher_action_max_abs=float((reproduced - upstream).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--release-unused-cache', action='store_true',
                        help='Release inactive allocator blocks before backward on a shared/busy GPU; no math changes.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output).resolve()
    if (output / 'result.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed smoke: {output}')
    assert cfg['training']['action_consistency']['video_conditioning'] == 'one_pass_future_cache'
    cfg['data']['train'].update(batch_size=1, num_workers=0)
    cfg['data']['val'].update(batch_size=1, num_workers=0)
    cfg['training'].update(max_train_iters=args.steps, save_every_iters=0, save_final=True,
                           output_dir=str(output), lr_warmup_iters=0)
    cfg['logging']['wandb']['enable'] = False
    cfg['logging']['train_log_every_iters'] = 1
    cfg['inference'].update(infer_every_iters=args.steps, num_samples=1)
    cfg['resume'] = {'auto_resume': False, 'resume_ckpt_path': None}
    init_distributed(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        prepare_data(cfg)
        if dist.get_rank() == 0:
            (output / 'smoke_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        model = build_model(cfg)
        model.load_components()
        train_data = build_data(cfg, train_or_val='train')
        val_data = build_data(cfg, train_or_val='val')
        trainer = build_trainer(cfg)
        trainer.set_model(model)
        trainer.set_data(train_data, val_data)
        parity = check_parity(model, next(iter(val_data)), trainer)
        print(f'PARITY_OK rank={dist.get_rank()} {json.dumps(parity)}', flush=True)
        torch.cuda.empty_cache()
        if args.release_unused_cache:
            original_loss = trainer._loss
            def release_after_forward(*positional, **keywords):
                result = original_loss(*positional, **keywords)
                torch.cuda.empty_cache()
                return result
            trainer._loss = release_after_forward
        faulthandler.dump_traceback_later(120, repeat=True)
        trainer.train()
        for parameter in trainer.student_params:
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            assert bool(torch.isfinite(parameter).all())
        for frozen in (trainer.roles.teacher, trainer.roles.target, model.unwrap_module().video_expert):
            assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
        assert any(not torch.equal(target, student) for target, student in trainer.roles.ema_pairs)
        student_hash = digest(trainer.student_params)
        ema_hash = digest(target for target, _ in trainer.roles.ema_pairs)
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, (student_hash, ema_hash))
        assert len(set(states)) == 1, states
        # Perturb then restore to test the actual per-rank checkpoint load path.
        with torch.no_grad():
            trainer.student_params[0].add_(1)
        restored_step = trainer.checkpoints.load(str(output / f'checkpoint-{args.steps:09d}'))
        assert restored_step == args.steps and digest(trainer.student_params) == student_hash
        assert digest(target for target, _ in trainer.roles.ema_pairs) == ema_hash
        rank_result = dict(rank=dist.get_rank(), parity=parity,
                           peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3)
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, rank_result)
        if dist.get_rank() == 0:
            result = dict(config=str(Path(args.config).resolve()),
                          video_conditioning=trainer.parsed.video_conditioning,
                          flow_target=trainer.parsed.flow_target,
                          flow_loss_weight=trainer.parsed.flow_loss_weight,
                          world_size=dist.get_world_size(), steps=args.steps, batch_per_rank=1,
                          teacher_steps=trainer.parsed.teacher_reference_steps,
                          all_gradients_finite=True, frozen_roles_no_grad=True,
                          student_and_ema_identical_across_ranks=True, checkpoint_roundtrip=True,
                          trainable_tensors=len(trainer.student_params), student_sha256=student_hash,
                          release_unused_cache=args.release_unused_cache,
                          ranks=ranks, scope='Real-data mechanical smoke, not a task success-rate evaluation.')
            (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print('FUTURE_CACHE_SMOKE_DONE', json.dumps(result), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        cleanup_distributed()


if __name__ == '__main__':
    main()
