"""Online contracts tested with real autograd/AdamW and injected deterministic I/O."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from grpo import train
from grpo.io import read_json


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_state = nn.Parameter(torch.tensor([.1, -.2, .3]))
        self.backbone = nn.Linear(3, 3, bias=False)
        self.backbone.requires_grad_(False)

    def forward(self, ids, targets, chunk_size=128):
        logits = self.initial_state * (ids.float()[..., None] + 1)
        return logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)


def fake_rollout(policy, prompts, tokenizer, config, seeds):
    results = []
    for i, (p, seed) in enumerate(zip(prompts, seeds)):
        completion = [1] if i % 2 == 0 else [2, 1]
        seq = p + completion
        with torch.no_grad():
            old = policy(torch.tensor([seq[:-1]]), torch.tensor([seq[1:]]))[0, len(p) - 1:].tolist()
        results.append(dict(completion_ids=completion, completion=str(completion[0]),
                            old_logprobs=old, mask=[1] * len(completion), seed=seed, finish_reason='length'))
    return results


class FakeJudge:
    def __init__(self):
        self.calls = []
        self.fail = False

    def score(self, p, completion, reference):
        self.calls.append((p, completion))
        if self.fail and completion == '2':
            raise RuntimeError('injected judge failure')
        value = float(completion)
        return {'reward': value / 4, 'scores': {**{k: value for k in ('semantic', 'instruction', 'grounding', 'task')},
                                             'reason': 'fixture'}, 'prior_errors': []}


class FakeLogger:
    def __init__(self):
        self.logs = []
        self.runs = []
        self.fail = False
    def init(self, **kw):
        self.runs.append(kw['id'])
    def Text(self, content, caption):
        return content
    def log(self, metrics, step):
        if self.fail:
            raise RuntimeError('injected upload failure')
        self.logs.append((step, metrics))
    def finish(self):
        pass


def config(tmp_path, **kw):
    c = dict(output=str(tmp_path), prompts_per_step=2, group_size=2, microbatch=1,
             epochs=1, learning_rate=.01, progress=False, sampling={'max_new_tokens': 2},
             activation_checkpointing=False, judge={}, shuffle=False)
    c.update(kw)
    train.validate(c)
    return c


def rows(n=6):
    return [dict(id=str(i), prompt=f'prompt {i}', prompt_ids=[i % 3], reference='reference') for i in range(n)]


def loop(c, policy, judge=None, logger=None, selected=None):
    return train.online_loop(c, policy, {'backbone_sha256': 'fixture'}, selected or rows(), None, judge or FakeJudge(), logger)


def test_three_steps_fresh_policy_complete_trace(tmp_path, monkeypatch):
    seen = []
    def rollout(policy, *args):
        seen.append(policy.initial_state.detach().clone())
        return fake_rollout(policy, *args)
    monkeypatch.setattr(train, 'rollout_batch', rollout)
    policy, logger = TinyPolicy(), FakeLogger()
    weights = policy.backbone.weight.detach().clone()
    steps = []
    original = torch.optim.AdamW.step
    def update(opt, *a, **kw):
        steps.append(1)
        return original(opt, *a, **kw)
    monkeypatch.setattr(torch.optim.AdamW, 'step', update)
    ck = loop(config(tmp_path), policy, logger=logger)
    assert ck['global_step'] == ck['policy_version'] == len(steps) == 3
    assert len(seen) == 3 and all(not torch.equal(a, b) for a, b in zip(seen, seen[1:]))
    assert torch.equal(weights, policy.backbone.weight) and policy.backbone.weight.grad is None
    assert policy.initial_state.dtype == torch.float32
    for step, metric in logger.logs:
        trace = [json.loads(line) for line in ''.join(metric['trace/samples']).splitlines()]
        assert len(trace) == 4 and {s['policy_version'] for s in trace} == {step - 1}
        assert {s['step_id'] for s in trace} == {step}
        assert all('old_logprobs' in s and 'judge' in s and 'advantage' in s for s in trace)
        assert metric['training/valid_tokens'] == 6 and metric['training/ratio_mean'] == pytest.approx(1.)
        if step < 3:
            state = torch.load(tmp_path / f'state-{step:08d}.pt', weights_only=True)['initial_state']
            torch.testing.assert_close(state, seen[step])


def test_microbatch_global_normalization(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'rollout_batch', fake_rollout)
    a, b = TinyPolicy(), TinyPolicy()
    ca = loop(config(tmp_path / 'a', microbatch=1), a)
    cb = loop(config(tmp_path / 'b', microbatch=8), b)
    torch.testing.assert_close(a.initial_state, b.initial_state)
    for key in ['exp_avg', 'exp_avg_sq']:
        torch.testing.assert_close(ca['optimizer']['state'][0][key], cb['optimizer']['state'][0][key])


@pytest.mark.parametrize('phase', ['rollout', 'judge', 'training', 'commit', 'logging'])
def test_interrupt_resume_exactly_once(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(train, 'rollout_batch', fake_rollout)
    reference = TinyPolicy()
    expected = loop(config(tmp_path / 'reference'), reference)
    policy, judge, logger = TinyPolicy(), FakeJudge(), FakeLogger()
    c = config(tmp_path / 'resume')
    if phase == 'rollout':
        monkeypatch.setattr(train, 'rollout_batch', lambda *a: (_ for _ in ()).throw(RuntimeError('injected')))
    if phase == 'judge':
        judge.fail = True
    original_step = torch.optim.AdamW.step
    if phase == 'training':
        def fail(opt, *args, **kw):
            original_step(opt, *args, **kw)
            raise RuntimeError('injected after optimizer step')
        monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    original_save = train.atomic_torch
    if phase == 'commit':
        def fail(path, obj):
            if obj['global_step'] > 0:
                raise RuntimeError('injected before commit')
            return original_save(path, obj)
        monkeypatch.setattr(train, 'atomic_torch', fail)
    if phase == 'logging':
        logger.fail = True
    with pytest.raises(RuntimeError):
        loop(c, policy, judge, logger)
    judge.fail = logger.fail = False
    monkeypatch.setattr(train, 'rollout_batch', fake_rollout)
    monkeypatch.setattr(torch.optim.AdamW, 'step', original_step)
    monkeypatch.setattr(train, 'atomic_torch', original_save)
    resumed = loop({**c, 'resume': True}, TinyPolicy(), judge, logger)
    assert resumed['global_step'] == resumed['policy_version'] == 3
    torch.testing.assert_close(resumed['initial_state'], expected['initial_state'])
    for key in ['exp_avg', 'exp_avg_sq', 'step']:
        torch.testing.assert_close(resumed['optimizer']['state'][0][key], expected['optimizer']['state'][0][key])
    assert len(set(logger.runs)) == 1
    assert [step for step, _ in logger.logs] == [1, 2, 3]
    assert not (Path(c['output']) / 'pending.json').exists()
    if phase == 'judge':
        assert judge.calls.count(('prompt 0', '1')) == 1  # successful request reused


def test_epochs_tail_zero_advantage_and_stale_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'rollout_batch', fake_rollout)
    class ConstantJudge(FakeJudge):
        def score(self, *a):
            return super().score(a[0], '1', a[2])
    policy = TinyPolicy()
    before = policy.initial_state.detach().clone()
    ck = loop(config(tmp_path, epochs=2, prompts_per_step=2), policy, ConstantJudge(), selected=rows(3))
    assert ck['global_step'] == 4 and ck['epoch'] == 2 and ck['policy_version'] == 0
    assert not ck['optimizer']['state']
    torch.testing.assert_close(policy.initial_state, before)
    with pytest.raises(ValueError, match='requires resume'):
        loop(config(tmp_path, epochs=2, prompts_per_step=2), TinyPolicy(), selected=rows(3))


def test_pending_policy_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'rollout_batch', fake_rollout)
    judge = FakeJudge()
    judge.fail = True
    c = config(tmp_path)
    with pytest.raises(RuntimeError):
        loop(c, TinyPolicy(), judge)
    pending = read_json(tmp_path / 'pending.json')
    pending['contract']['policy_version'] = 99
    train.atomic_json(tmp_path / 'pending.json', pending)
    with pytest.raises(ValueError, match='different policy'):
        loop({**c, 'resume': True}, TinyPolicy())


def test_large_trace_is_not_truncated_by_swanlab_media_limit(tmp_path):
    (tmp_path / 'steps').mkdir()
    content = '完整记录' * 1000
    (tmp_path / 'steps/00000001.jsonl').write_text(content)
    logger = FakeLogger()
    train.log_step(logger, tmp_path, 1, {'training/loss': 1.}, text_bytes=32)
    payload = logger.logs[0][1]
    parts = [value for key, value in payload.items() if key.startswith('trace/samples')]
    assert all(len(part) <= 100 for part in parts)
    assert ''.join(chunk for part in parts for chunk in part) == content


def test_alignment_failure_never_commits_or_updates(tmp_path, monkeypatch):
    def corrupt(*args):
        samples = fake_rollout(*args)
        samples[0]['old_logprobs'][0] += 1
        return samples
    monkeypatch.setattr(train, 'rollout_batch', corrupt)
    steps = []
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *args: steps.append(1))
    with pytest.raises(RuntimeError, match='alignment'):
        loop(config(tmp_path), TinyPolicy())
    ck = torch.load(tmp_path / 'latest.pt', weights_only=True)
    assert ck['global_step'] == ck['policy_version'] == 0
    assert not steps


def test_resume_restores_sampling_rng(tmp_path, monkeypatch):
    import random
    def stochastic(*args):
        random.random()
        torch.rand(5)
        return fake_rollout(*args)
    monkeypatch.setattr(train, 'rollout_batch', stochastic)
    random.seed(777)
    torch.manual_seed(777)
    expected = loop(config(tmp_path / 'continuous'), TinyPolicy())
    random.seed(777)
    torch.manual_seed(777)
    c = config(tmp_path / 'resumed')
    loop({**c, 'stop_after_steps': 1}, TinyPolicy())
    random.random()
    torch.rand(100)
    resumed = loop({**c, 'resume': True}, TinyPolicy())
    assert resumed['rng'][0]['python'] == expected['rng'][0]['python']
    torch.testing.assert_close(resumed['rng'][0]['torch'], expected['rng'][0]['torch'])


def alignment_batch(policy, count, drift_count, drift):
    with torch.no_grad():
        old = float(policy(torch.tensor([[0]]), torch.tensor([[1]]))[0, 0])
    return [dict(prompt_ids=[0], completion_ids=[1], mask=[1],
                 old_logprobs=[old - (drift if i < drift_count else 0)],
                 advantage=(-1.) ** i) for i in range(count)]


def test_isolated_bf16_outlier_is_diagnostic_not_step_failure():
    from grpo.online import train_step
    policy = TinyPolicy()
    optimizer = torch.optim.AdamW([policy.initial_state], lr=.001, weight_decay=0)
    samples = alignment_batch(policy, 1000, 1, .578527)
    c = dict(microbatch=128, group_size=2, alignment_tolerance=.3,
             alignment_mean_ratio_tolerance=.02, alignment_max_clip_fraction=.01)
    metrics = train_step(policy, policy, optimizer, samples, c)
    assert metrics['alignment/max_error_exceeded'] == 1
    assert metrics['alignment/max_abs_logprob_error'] == pytest.approx(.578527)
    assert metrics['training/clip_fraction'] == pytest.approx(.001)
    assert optimizer.state[policy.initial_state]['step'] == 1


@pytest.mark.parametrize('drift_count,drift', [(1000, .05), (20, .25)])
def test_population_alignment_failure_still_prevents_update(drift_count, drift):
    from grpo.online import train_step
    policy = TinyPolicy()
    before = policy.initial_state.detach().clone()
    optimizer = torch.optim.AdamW([policy.initial_state], lr=.001, weight_decay=0)
    samples = alignment_batch(policy, 1000, drift_count, drift)
    with pytest.raises(RuntimeError, match='ratio alignment'):
        train_step(policy, policy, optimizer, samples, dict(microbatch=128, group_size=2))
    torch.testing.assert_close(policy.initial_state, before)
    assert not optimizer.state


def test_resume_cli_keeps_original_training_config(tmp_path, monkeypatch):
    from grpo.io import config_cli
    path = tmp_path / 'config.json'
    original = dict(output='run', resume=False, microbatch=16)
    path.write_text(json.dumps(original))
    monkeypatch.setattr('sys.argv', ['grpo.train', '--config', str(path), '--resume', '--stop-after-steps', '3'])
    c = config_cli('online', resume_options=True)
    assert c == {**original, 'resume': True, 'stop_after_steps': 3}
    assert read_json(path) == original
