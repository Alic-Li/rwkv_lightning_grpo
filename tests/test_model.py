import os
from pathlib import Path

import pytest
import torch

from grpo.data import collate, ppo_sum
from grpo.model import Backbone, StatePolicy
from grpo.rollout import rollout, rollout_batch
from grpo.tokenizer import TRIE_TOKENIZER

MODEL = os.environ.get(
    "GRPO_TEST_MODEL", "/mnt/sda1/rwkv_weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth"
)
VOCAB = os.environ.get(
    "GRPO_TEST_VOCAB",
    "/home/alic-li/RWKV-LM-V7/data/tokenizer/rwkv_vocab_v20230424.txt",
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Path(MODEL).exists(),
    reason="Local RWKV checkpoint and GPU required",
)


@pytest.fixture(scope="module")
def policy():
    b = Backbone(MODEL)
    return StatePolicy(b, torch.zeros(b.n_layer, b.n_head, 64, 64))


def test_freeze_graph_update(policy):
    tokenizer = TRIE_TOKENIZER(VOCAB)
    p = tokenizer.encode("User: What is 2+2?\n\nAssistant:")
    rows = [
        {
            "prompt_ids": p,
            "completion_ids": tokenizer.encode(" 4") + [0],
            "old_logprobs": [0.0, 0.0],
            "mask": [1, 1],
            "advantage": 1.0,
        }
    ]
    rows[0]["old_logprobs"] = [0.0] * len(rows[0]["completion_ids"])
    rows[0]["mask"] = [1] * len(rows[0]["completion_ids"])
    ids, target, old, mask, adv = collate(rows, "cuda")
    base = policy.initial_state.detach().clone()
    weights = [w.detach().clone() for w in policy.backbone.parameters()]
    policy.zero_grad(set_to_none=True)
    new = policy(ids, target, chunk_size=7)
    old = new.detach()
    loss, _, _ = ppo_sum(new, old, adv, mask, 0.2)
    (loss / mask.sum()).backward()
    grad = policy.initial_state.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    assert all(
        w.grad is None and not w.requires_grad and w.dtype == torch.bfloat16
        for w in policy.backbone.parameters()
    )
    assert policy.initial_state.dtype == torch.float32
    # Completion-only loss must reach initial state THROUGH prompt recurrence.
    assert all(grad[i].norm() > 0 for i in range(len(grad)))
    torch.optim.AdamW([policy.initial_state], lr=1e-4, weight_decay=0).step()
    assert not torch.equal(base, policy.initial_state)
    for old, w in zip(weights, policy.backbone.parameters()):
        assert torch.equal(old, w)
    with torch.no_grad():
        policy.initial_state.copy_(base)


def test_isolation_and_teacher_forcing(policy):
    t = TRIE_TOKENIZER(VOCAB)
    p = t.encode("User: Hi\n\nAssistant:")
    cfg = {"max_new_tokens": 5, "prefill_chunk": 128, "eos_id": 0, "stop_strings": []}
    before = policy.initial_state.detach().clone()
    a = rollout(policy, p, t, cfg, 77)
    b = rollout(policy, p, t, cfg, 77)
    assert a == b and torch.equal(before, policy.initial_state)
    s = {**a, "prompt_ids": p, "advantage": 1}
    ids, target, old, mask, _ = collate([s], "cuda")
    with torch.no_grad():
        new = policy(ids, target, 128)
    error = (new - old)[mask.bool()].abs().max()
    assert error < 0.15
    assert torch.all((new - old)[mask.bool()].exp().sub(1).abs() < 0.16)


def test_padding_valid_logits(policy):
    rows = [
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4, 0],
            "old_logprobs": [0, 0],
            "mask": [1, 1],
            "advantage": 1,
        },
        {
            "prompt_ids": [2],
            "completion_ids": [0],
            "old_logprobs": [0],
            "mask": [1],
            "advantage": -1,
        },
    ]
    ids, targets, _, mask, _ = collate(rows, "cuda")
    with torch.no_grad():
        together = policy(ids, targets, 128)
        for i, s in enumerate(rows):
            si, st, _, sm, _ = collate([s], "cuda")
            single = policy(si, st, 128)
            torch.testing.assert_close(
                together[i][mask[i].bool()], single[sm.bool()], atol=0.15, rtol=0.02
            )


@pytest.mark.parametrize("prompt_count", [4, 8])
def test_gpu_parallel_rollout_alignment(policy, prompt_count):
    tokenizer = TRIE_TOKENIZER(VOCAB)
    prompts = [
        tokenizer.encode(f"User: What is {i}+2? Please explain.\n\nAssistant:")
        for i in range(prompt_count)
        for _ in range(8)
    ]
    shapes = []
    hook = policy.backbone.register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape))
    )
    before = policy.initial_state.detach().clone()
    try:
        samples = rollout_batch(
            policy,
            prompts,
            tokenizer,
            {
                "max_new_tokens": 8,
                "prefill_chunk": 128,
                "eos_id": -1,
                "stop_strings": [],
            },
            list(range(len(prompts))),
        )
    finally:
        hook.remove()
    assert (prompt_count * 8, 1) in shapes
    assert torch.equal(before, policy.initial_state)
    for i in (0, len(samples) // 2, len(samples) - 1):
        sample = {**samples[i], "prompt_ids": prompts[i], "advantage": 1.0}
        ids, targets, old, mask, _ = collate([sample], "cuda")
        with torch.no_grad():
            new = policy(ids, targets, 128)
        delta = (new - old)[mask.bool()]
        assert delta.abs().max() < 0.15
        assert abs(delta.exp().mean().item() - 1) < 0.05


def test_checkpoint_microbatch_preserves_full_state_gradient(policy):
    from grpo.training import TrainingPolicy

    rows = []
    for i in range(16):
        rows.append(
            {
                "prompt_ids": [1, 2, 3, 4] * (1 + i % 3),
                "completion_ids": [5, 6, 0],
                "old_logprobs": [-2.0] * 3,
                "mask": [1, 1, 1],
                "advantage": (-1.0) ** i,
            }
        )
    ids, targets, old, mask, adv = collate(rows, "cuda")
    shapes = []
    hook = policy.backbone.register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape))
    )
    results = []
    try:
        for use_checkpoint in (False, True):
            policy.zero_grad(set_to_none=True)
            model = TrainingPolicy(policy, use_checkpoint)
            new = model(ids, targets, 7)
            # Signed completion-only loss traverses all preceding prompt chunks.
            loss = -(new * mask * adv).sum() / mask.sum()
            loss.backward()
            results.append((new.detach(), policy.initial_state.grad.detach().clone()))
    finally:
        hook.remove()
    assert all(b == 16 for b, t in shapes)
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-5, atol=1e-6)
    grad = results[1][1]
    assert torch.isfinite(grad).all() and all(g.norm() > 0 for g in grad)
    assert all(p.grad is None for p in policy.backbone.parameters())
    assert [p for p in model.parameters() if p.requires_grad] == [policy.initial_state]


def test_completion_only_logsoftmax_keeps_full_prompt_bptt(policy):
    from grpo.training import TrainingPolicy
    samples = [dict(prompt_ids=[1, 2, 3] * n, completion_ids=[4, 5, 0],
                    old_logprobs=[0.] * 3, mask=[1] * 3, advantage=(-1.) ** i)
               for i, n in enumerate([12, 2])]
    ids, targets, old, mask, adv = collate(samples, 'cuda')
    results = []
    model = TrainingPolicy(policy, True)
    for selected in [None, mask]:
        policy.zero_grad(set_to_none=True)
        new = model(ids, targets, 7, completion_mask=selected)
        loss = -(new * mask * adv).sum() / mask.sum()
        loss.backward()
        results.append((new.detach()[mask.bool()], policy.initial_state.grad.detach().clone()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-5, atol=1e-6)
    assert all(s.norm() > 0 for s in results[1][1])
    assert all(p.grad is None for p in policy.backbone.parameters())
