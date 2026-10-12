import json

import pytest
import torch

from grpo.data import advantages, collate, ppo_sum, prompts


def test_group_advantages():
    _, _, a = advantages([1.0, 3.0])
    _, _, b = advantages([100.0, 102.0])
    _, s, z = advantages([4.0, 4.0, 4.0])
    assert a == pytest.approx(b) and a == pytest.approx([-1, 1], abs=1e-5)
    assert s == 0 and z == [0, 0, 0]
    with pytest.raises(ValueError):
        advantages([1, float("nan")])


def test_prompt_semantics(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"text": "U:", "train": False},
                    {"text": "A", "train": True},
                    {"text": "T:", "train": False},
                    {"text": "B\n\nUser:", "train": True},
                ]
            }
        )
    )

    class Tokenizer:
        def encode(self, s):
            return list(s.encode())

    rows = list(prompts(path, Tokenizer(), 100))
    assert [r["prompt"] for r in rows] == ["U:", "U:AT:"]
    assert rows[1]["reference"] == "B\n\nUser:"
    assert list(prompts(path, Tokenizer(), 2)) == rows[:1]


def test_shift_eos_padding_and_signs():
    rows = [
        {
            "prompt_ids": [2, 3],
            "completion_ids": [4, 0],
            "old_logprobs": [-1, -2],
            "mask": [1, 1],
            "advantage": 1,
        },
        {
            "prompt_ids": [7],
            "completion_ids": [8],
            "old_logprobs": [-3],
            "mask": [1],
            "advantage": -1,
        },
    ]
    ids, target, old, mask, adv = collate(rows, "cpu")
    assert ids.tolist() == [[2, 3, 4], [7, 0, 0]]
    assert target.tolist() == [[3, 4, 0], [8, 0, 0]]
    assert mask.tolist() == [[0, 1, 1], [1, 0, 0]]
    new = old.clone().requires_grad_()
    loss, ratio, _ = ppo_sum(new, old, adv, mask, 0.2)
    loss.backward()
    assert new.grad.tolist() == [[0, -1, -1], [1, 0, 0]]
    assert ratio == 3
    r = torch.tensor([[1.4, 0.6, 1.4, 0.6]])
    a = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    loss, _, cf = ppo_sum(r.log(), torch.zeros_like(r), a, torch.ones_like(r), 0.2)
    assert loss.item() == pytest.approx(-1.2 - 0.6 + 1.4 + 0.8)
    assert cf == 4
    new = torch.tensor([[1000.0, 0.0]], requires_grad=True)
    l, _, _ = ppo_sum(
        new, torch.zeros_like(new), torch.ones(1, 1), torch.tensor([[0.0, 1.0]]), 0.2
    )
    assert torch.isfinite(l)


def test_rollout_eos_and_reserved_ids():
    from grpo.rollout import decode_generated, rollout

    class Tokenizer:
        idx2token = {1: b"x", 2: b" answer", 3: b"\n\nUser:extra"}

    class Policy:
        initial_state = torch.zeros(1)

        def __init__(self, tokens):
            self.tokens = iter(tokens)
            self.calls = 0

        def start(self, b):
            return torch.zeros(1, b, 1, 1, 1)

        def backbone(self, *args, **kwargs):
            assert not torch.is_grad_enabled()
            self.calls += 1
            logits = torch.full((1, 1, 6), float("-inf"))
            logits[0, 0, next(self.tokens)] = 0
            return logits, torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 2, 1, 1)

    p = Policy([2, 0, 1])
    s = rollout(p, [1], Tokenizer(), {"max_new_tokens": 8}, 1)
    assert s["completion_ids"] == [2, 0] and s["mask"] == [1, 1]
    assert (
        s["old_logprobs"] == [0.0, 0.0] and s["finish_reason"] == "eos" and p.calls == 2
    )
    p = Policy([3, 1])
    s = rollout(p, [1], Tokenizer(), {"max_new_tokens": 8}, 1)
    assert s["finish_reason"] == "stop" and s["completion_ids"] == [3] and p.calls == 1
    assert decode_generated(Tokenizer(), [5]) == "<|token:5|>"
