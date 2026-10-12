"""Batch boundaries, per-row RNG, early stop and state isolation."""

import pytest
import torch

from grpo.rollout import prefill_batch, rollout, rollout_batch


class Tokenizer:
    idx2token = {1: b"a", 2: b"b", 3: b"c", 4: b"d", 5: b"e"}


class ToyPolicy:
    def __init__(self, deterministic=False):
        self.initial_state = torch.nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.calls = []
        self.deterministic = deterministic

    def start(self, batch):
        return self.initial_state[:, None].expand(-1, batch, -1, -1, -1).contiguous()

    def backbone(self, ids, states, shifts=None, last_only=True):
        assert not torch.is_grad_enabled()
        self.calls.append(tuple(ids.shape))
        states = states.clone()
        for pos in range(ids.shape[1]):
            states = (states * 2 + ids[None, :, pos, None, None, None] + 1) % 7
        center = states.reshape(-1).long() % 6
        if self.deterministic:
            logits = torch.full((len(ids), 6), float("-inf"))
            logits.scatter_(1, center[:, None], 0)
        else:
            logits = -(torch.arange(6)[None, :] - center[:, None]).float().square() / 3
        shifts = ids[:, -1].float()[None, None, :, None].expand(1, 2, -1, 1).clone()
        return logits[:, None], states, shifts


def test_short_prompt_tail_does_not_change_long_prompt_chunk_grid():
    policy = ToyPolicy()
    prefill_batch(policy, [[1, 2, 3], [1, 2, 3, 4, 5, 1]], 2)
    assert policy.calls == [(2, 2), (1, 1), (1, 2), (1, 2)]


@pytest.mark.parametrize("batch", [32, 64])
def test_true_batch_decode_and_independent_early_stops(batch):
    p = ToyPolicy(deterministic=True)
    prompts = [[1], [2, 3], [4, 2, 2], [3, 3]]
    inputs = [prompts[i % 4] for i in range(batch)]
    completed = []
    result = rollout_batch(
        p,
        inputs,
        Tokenizer(),
        {"max_new_tokens": 8, "prefill_chunk": 2},
        list(range(batch)),
        on_complete=lambda i, s: completed.append(i),
    )
    assert (
        batch,
        1,
    ) in p.calls  # Actual multi-row decode, not a loop of single-row forwards.
    assert len(completed) == batch and sorted(completed) == list(range(batch))
    assert len(set(len(s["completion_ids"]) for s in result)) > 1
    assert all(
        len(s["mask"]) == len(s["old_logprobs"]) == len(s["completion_ids"])
        for s in result
    )
    assert all(s["mask"] == [1] * len(s["completion_ids"]) for s in result)
    assert all(s["old_logprobs"] == [0.0] * len(s["completion_ids"]) for s in result)
    assert p.initial_state.grad is None and torch.count_nonzero(p.initial_state) == 0


def test_mixed_prompt_lengths_match_individual_rollouts_and_reordered_batch():
    prompts = [[1], [2, 3, 1], [1], [4, 3, 2, 1, 4], [2, 3, 1]]
    seeds = [80, 90, 91, 42, 123]
    config = {"max_new_tokens": 12, "prefill_chunk": 2, "stop_strings": ["be"]}
    batch = rollout_batch(ToyPolicy(), prompts, Tokenizer(), config, seeds)
    reverse = rollout_batch(
        ToyPolicy(), prompts[::-1], Tokenizer(), config, seeds[::-1]
    )[::-1]
    for i, s in enumerate(batch):
        single = rollout(ToyPolicy(), prompts[i], Tokenizer(), config, seeds[i])
        for key in ("completion_ids", "old_logprobs", "mask", "finish_reason", "seed"):
            assert s[key] == single[key] == reverse[i][key]


def test_completed_trajectory_callback_survives_interrupt():
    saved = {}

    def save(i, s):
        saved[i] = s
        raise RuntimeError("interrupt after atomic save")

    with pytest.raises(RuntimeError, match="interrupt"):
        rollout_batch(
            ToyPolicy(True),
            [[1], [2, 3]],
            Tokenizer(),
            {"max_new_tokens": 8},
            [1, 2],
            on_complete=save,
        )
    assert len(saved) == 1
