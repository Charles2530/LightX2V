from types import SimpleNamespace

import pytest
import torch

from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy


class FakeTokenizer:
    def __call__(self, texts, return_mask, add_special_tokens):
        assert return_mask and add_special_tokens and len(texts) == 1
        return torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]])


class FakeTextModel:
    def __call__(self, ids, mask):
        return torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)


def make_policy():
    policy = FastWAMPolicy.__new__(FastWAMPolicy)
    policy.default_prompt = "Task: {task_prompt}"
    policy.device = torch.device("cpu")
    policy._prompt_cache = {}
    policy.text_encoder_released = False
    policy.text_encoder = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeTextModel())
    return policy


def test_preloads_unique_prompts_and_releases_encoder():
    policy = make_policy()

    cache_size = policy.preload_task_prompts(["pick", "pick", "place"], release_text_encoder=True)

    assert cache_size == 2
    assert policy.text_encoder is None
    assert policy.text_encoder_released is True
    context, mask = policy.encode_prompt("Task: pick")
    assert context.shape == (3, 6)
    assert mask.tolist() == [True, True, True]
    assert torch.count_nonzero(context[2]) == 0

    with pytest.raises(RuntimeError, match="prompt was not cached"):
        policy.encode_prompt("Task: unseen")
