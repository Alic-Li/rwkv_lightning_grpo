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
        state = torch.zeros(1, 1, 64, 64)
        state.flatten()[:3] = torch.tensor([.1, -.2, .3])
        self.initial_state = nn.Parameter(state)
        self.backbone = nn.Linear(3, 3, bias=False)
        self.backbone.requires_grad_(False)

    def forward(self, ids, targets, chunk_size=128, completion_mask=None):
        logits = self.initial_state.flatten()[:3] * (ids.float()[..., None] + 1)
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
             activation_checkpointing=False, judge={}, shuffle=False, judge_retry_rounds=1)
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
        exported = torch.load(tmp_path / f'state-{step:08d}.pth', weights_only=True)
        full = torch.load(tmp_path / f'state-{step:08d}.pt', weights_only=True)
        assert torch.equal(exported['blocks.0.att.time_state'], full['initial_state'][0].bfloat16())
        trace = [json.loads(line) for line in (tmp_path / f'steps/{step:08d}.jsonl').read_text().splitlines()]
        assert set(metric) <= train.SWANLAB_METRICS
        assert not any(k.startswith(('performance/', 'trace/', 'reward/group/')) for k in metric)
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


@pytest.mark.parametrize('phase', ['rollout', 'judge', 'training', 'commit', 'export', 'logging'])
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
            if obj.get('global_step', 0) > 0:
                raise RuntimeError('injected before commit')
            return original_save(path, obj)
        monkeypatch.setattr(train, 'atomic_torch', fail)
    if phase == 'export':
        def fail(path, obj):
            if path.name == 'latest-state.pth':
                committed = torch.load(Path(c['output']) / 'latest.pt', weights_only=True)
                if committed['global_step'] > 0:
                    raise RuntimeError('injected after durable commit before export')
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
    exported = torch.load(Path(c['output']) / 'latest-state.pth', weights_only=True)
    assert set(exported) == {'blocks.0.att.time_state'}
    assert exported['blocks.0.att.time_state'].dtype == torch.bfloat16
    assert torch.equal(exported['blocks.0.att.time_state'], resumed['initial_state'][0].bfloat16())
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


def test_swanlab_core_metrics_do_not_upload_detail_or_read_trace(tmp_path):
    logger = FakeLogger()
    metrics = {
        'training/loss': 1., 'reward/mean': .5, 'judge/success_rate': 1.,
        'performance/step_seconds': 10., 'trace/samples': 'private detail',
        'reward/group/000168-005': .7, 'judge/semantic/mean': 3.,
        'advantage/std': 1., 'alignment/max_abs_logprob_error': .3,
        'rollout/stop/eos': .5,
    }
    # No steps directory: logging must not open or upload sample files.
    train.log_step(logger, tmp_path, 1, metrics)
    assert logger.logs == [(1, {'training/loss': 1., 'reward/mean': .5, 'judge/success_rate': 1.})]
    assert read_json(tmp_path / 'logged.json') == {'step': 1}


def test_alignment_failure_never_commits_or_updates(tmp_path, monkeypatch):
    calls = []
    def corrupt(*args):
        samples = fake_rollout(*args)
        if not calls:
            samples[0]['old_logprobs'][0] += 1
        calls.append(1)
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


def test_length_bucketing_reduces_padding_with_same_update():
    from grpo.online import train_step
    samples = []
    a, b = TinyPolicy(), TinyPolicy()
    for i in range(24):
        p = [0] * (40 if (i // 2) % 2 else 1)
        c = [1, 1] if i % 2 else [2, 2]
        seq = p + c
        with torch.no_grad():
            old = a(torch.tensor([seq[:-1]]), torch.tensor([seq[1:]]))[0, len(p) - 1:].tolist()
        samples.append(dict(prompt_ids=p, completion_ids=c, old_logprobs=old,
                            mask=[1, 1], advantage=(-1.) ** i))
    metrics = []
    for policy, bucket in [(a, False), (b, True)]:
        opt = torch.optim.AdamW([policy.initial_state], lr=.001)
        metrics.append(train_step(policy, policy, opt, samples, dict(microbatch=8, group_size=2, length_bucketing=bucket)))
    assert metrics[1]['training/teacher_forcing_tokens'] < metrics[0]['training/teacher_forcing_tokens']
    assert metrics[1]['training/padding_fraction'] < metrics[0]['training/padding_fraction']
    assert metrics[0]['training/valid_tokens'] == metrics[1]['training/valid_tokens'] == 48
    torch.testing.assert_close(a.initial_state, b.initial_state)


def test_population_outlier_requires_successful_behavior_replay():
    from grpo.online import train_step
    policy = TinyPolicy()
    optimizer = torch.optim.AdamW([policy.initial_state], lr=.001)
    samples = alignment_batch(policy, 1000, 21, .25)
    calls = []
    metrics = train_step(policy, policy, optimizer, samples, dict(microbatch=128, group_size=2),
                         verify_rollout=lambda: calls.append(1) or 0.)
    assert calls == [1] and metrics['alignment/behavior_replayed'] == 1
    assert metrics['training/clip_fraction'] == pytest.approx(.021)
    assert optimizer.state[policy.initial_state]['step'] == 1


@pytest.mark.parametrize('field', ['completion_ids', 'mask', 'old_logprobs'])
def test_behavior_replay_rejects_corrupted_sample(monkeypatch, field):
    monkeypatch.setattr(train, 'rollout_batch', fake_rollout)
    policy = TinyPolicy()
    generated = fake_rollout(policy, [[0], [0]], None, {}, [1, 2])
    samples = [dict(s, prompt_ids=[0], seed=i + 1) for i, s in enumerate(generated)]
    samples[0][field] = list(samples[0][field])
    samples[0][field][0] += 1
    with pytest.raises(RuntimeError, match='alignment verification failed'):
        train.verify_behavior_rollout(policy, samples, None, dict(group_size=2, sampling={}))


def test_behavior_replay_keeps_rank_group_assignment(monkeypatch):
    jobs = []
    def replay(policy, prompt_ids, tokenizer, sampling, seeds):
        jobs.append((prompt_ids, seeds))
        return [dict(completion_ids=[1], mask=[1], old_logprobs=[-1.]) for _ in seeds]
    monkeypatch.setattr(train, 'rollout_batch', replay)
    samples = [dict(prompt_ids=[i // 2], seed=i, completion_ids=[1], mask=[1], old_logprobs=[-1.]) for i in range(6)]
    for rank in range(2):
        assert train.verify_behavior_rollout(None, samples, None, dict(group_size=2, sampling={}), rank, 2) == 0
    assert jobs == [([[0], [0], [2], [2]], [0, 1, 4, 5]), ([[1], [1]], [2, 3])]
