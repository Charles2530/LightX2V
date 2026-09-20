"""Keep FasterWAM's deployment path separate from FastWAM's first-frame path."""
import inspect
from pathlib import Path
from types import ModuleType
import sys

import pytest

from lightx2v_train.infer.fasterwam_robotwin import create_fasterwam


def test_inference_uses_upstream_future_cache_and_preserves_signature(monkeypatch):
    class UpstreamModel:
        def infer_action(self, **kwargs):
            raise AssertionError("Joint rollout must not be used")

        def infer_action_one_pass_future_cache(self, *, num_video_frames, action_horizon, num_inference_steps):
            return (num_video_frames, action_horizon, num_inference_steps)

    seen = {}
    upstream = ModuleType('fasterwam.runtime')
    model = UpstreamModel()
    def factory(**kwargs):
        seen.update(kwargs)
        return model
    upstream.create_fasterwam = factory
    monkeypatch.setitem(sys.modules, 'fasterwam.runtime', upstream)
    # Inherited FastWAM config fields must not leak into the upstream constructor.
    built = create_fasterwam(compile_training_denoise=False, model_id='local', condition_layers=[0,4])
    assert seen == dict(model_id='local', condition_layers=[0,4])
    assert built.infer_action.__func__ is UpstreamModel.infer_action_one_pass_future_cache
    assert 'num_video_frames' in inspect.signature(built.infer_action).parameters
    assert built.infer_action(num_video_frames=9,action_horizon=32,num_inference_steps=1)==(9,32,1)


def test_training_compile_is_rejected():
    with pytest.raises(ValueError, match='compile_training_denoise'):
        create_fasterwam(compile_training_denoise=True)


def test_launchers_are_separate_and_export_lora():
    scripts = Path(__file__).parents[1]/'scripts'
    faster = (scripts/'eval_fasterwam_robotwin_future_kv_1step.sh').read_text()
    fast = (scripts/'eval_robotwin_distilled_1step.sh').read_text()
    assert 'eval_robotwin_distilled_1step.sh' not in faster
    assert 'robotwin_fasterwam_future_kv_1step' in faster
    assert 'EVALUATION.replan_steps=28' in faster
    assert '--lora-output' in faster and '--weights ema' in faster
    assert 'fasterwam' not in fast.lower()
