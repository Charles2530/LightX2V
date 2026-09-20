"""Compare the original FasterWAM, training adapter and exported EMA numerically."""
import argparse
import json
from pathlib import Path
from types import MethodType

import torch
import yaml
from torch.utils.data import default_collate

from fasterwam.models.wan22.fastwam import FastWAM
from lightx2v_train.data.robotwin_dataset import _build_robotwin_dataset
from lightx2v_train.model_zoo.wan_fasterwam import WanFasterWAMModel
from lightx2v_train.model_zoo.native.wan.fasterwam.action_distill import (
    ActionDistillCondition, CachedActionDenoiser, build_action_distill_condition,
    sample_action_teacher, sample_action_one_step,
)
from lightx2v_train.trainers.fasterwam_action_consistency.config import FastWAMActionConsistencyConfig
from lightx2v_train.trainers.fasterwam_action_consistency.roles import configure_student, load_role_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    checkpoint = root/'lightx2v_train/runs/fasterwam_robotwin_action_1step_consistency_ts10/checkpoint-000030000'
    cfg = yaml.safe_load((checkpoint/'config.yaml').read_text())
    result = {}

    def compare(name, a, b):
        diff = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
        result[name] = {'mae': float(diff.mean()), 'max_abs': float(diff.max())}
        print(name, result[name], flush=True)

    dataset = _build_robotwin_dataset(cfg['data']['val'], 'val')
    sample = default_collate([dataset[0]])
    print('SAMPLE', sample['prompt'], {k: tuple(v.shape) for k,v in sample.items() if isinstance(v,torch.Tensor)}, flush=True)
    result['prompt'] = sample['prompt'][0]
    cfg['model']['skip_dit_load_from_pretrain'] = True
    adapter = WanFasterWAMModel(cfg)
    adapter.load_components()
    model = adapter.unwrap_module().eval().requires_grad_(False)
    with torch.no_grad():
        inputs = model.build_inputs(sample)
        observation = build_action_distill_condition(model, inputs)
        noise = torch.randn(inputs['action'].shape, generator=torch.Generator().manual_seed(42)).to('cuda', torch.bfloat16)
        inference_kwargs = dict(prompt=None, input_image=sample['video'][:,:,0], action_horizon=32,
                                proprio=sample['proprio'][:,0], context=sample['context'],
                                context_mask=sample['context_mask'], seed=42, num_inference_steps=10)
        original_observation = FastWAM.infer_action(model, **inference_kwargs)['action']
        denoiser = CachedActionDenoiser(model.action_expert, model.mot)
        training_observation = sample_action_teacher(denoiser, noise, observation, model.infer_action_scheduler, 10)[0]
        compare('original_observation_vs_training_cache', original_observation, training_observation)

        captured = {}
        original_predict = model._predict_action_noise_with_cache
        def capture(self, **kwargs):
            if not captured:
                captured.update(kwargs)
            return original_predict(**kwargs)
        model._predict_action_noise_with_cache = MethodType(capture, model)
        original_future = model.infer_action_one_pass_future_cache(num_video_frames=9, **inference_kwargs)['action']
        model._predict_action_noise_with_cache = original_predict
        future = ActionDistillCondition(**{k: captured[k] for k in (
            'context','context_mask','video_kv_cache','attention_mask','video_seq_len')})
        result['observation_video_tokens'] = observation.video_seq_len
        result['original_future_video_tokens'] = future.video_seq_len
        print('CACHE_TOKENS', observation.video_seq_len, future.video_seq_len, flush=True)
        training_future = sample_action_teacher(denoiser, noise, future, model.infer_action_scheduler, 10)[0]
        compare('original_future_vs_adapter_same_cache', original_future, training_future)
        compare('released_observation_vs_future', original_observation, original_future)
        compare('released_observation_vs_gt', original_observation, inputs['action'][0])
        compare('released_future_vs_gt', original_future, inputs['action'][0])

    parsed = FastWAMActionConsistencyConfig.from_mapping(cfg)
    student = configure_student(model.action_expert, parsed.student)
    state = torch.load(checkpoint/'ema_action.pt', map_location='cpu', weights_only=True)
    load_role_state_dict(student, parsed.student.train_type, state)
    student.eval()
    ema = CachedActionDenoiser(student, model.mot)
    with torch.no_grad():
        ema_observation = sample_action_one_step(ema, noise, observation, 1000)[0]
        ema_future = sample_action_one_step(ema, noise, future, 1000)[0]
        compare('ema_observation_vs_released_observation', ema_observation, original_observation)
        compare('ema_future_vs_released_future', ema_future, original_future)
        compare('ema_observation_vs_gt', ema_observation, inputs['action'][0])
        compare('ema_future_vs_gt', ema_future, inputs['action'][0])

    # Verify real backward with SparseMoT's checkpointing and the saved LoRA.
    student.train()
    with adapter.autocast_context():
        loss = ema(noise, torch.full((1,),1000,device='cuda',dtype=torch.bfloat16), observation).float().square().mean()
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.requires_grad]
    result['gradient_parameters'] = len(grads)
    result['gradient_missing'] = sum(g is None for g in grads)
    result['gradient_nonfinite'] = sum(g is not None and not bool(torch.isfinite(g).all()) for g in grads)
    print('GRADIENT', {k:v for k,v in result.items() if k.startswith('gradient_')}, flush=True)
    student.zero_grad(set_to_none=True)

    student.eval()
    merged = student.merge_and_unload(safe_merge=True)
    model.action_expert = merged
    model.mot.mixtures['action'] = merged
    exported = torch.load(root/'fasterwam_robotwin_consistency_ts10_ema_step30000.pt',map_location='cpu',weights_only=False)
    current = model.mot.state_dict()
    assert set(current)==set(exported['mot'])
    mismatches = [k for k,v in current.items() if not torch.equal(v.cpu(),exported['mot'][k])]
    result['export_mismatched_tensors'] = mismatches
    assert not mismatches, mismatches
    assert all(torch.equal(v.cpu(), exported['proprio_encoder'][k]) for k,v in model.proprio_encoder.state_dict().items())
    print('ALL_EXPORT_TENSORS_EXACT', len(current), flush=True)
    with torch.no_grad():
        merged_denoiser = CachedActionDenoiser(merged, model.mot)
        merged_observation = sample_action_one_step(merged_denoiser, noise, observation,1000)[0]
        merged_future = sample_action_one_step(merged_denoiser, noise, future,1000)[0]
        compare('merged_vs_unmerged_observation', merged_observation, ema_observation)
        compare('merged_vs_unmerged_future', merged_future, ema_future)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('NUMERICS_DONE',args.output,flush=True)


if __name__ == '__main__':
    main()
