import json
from types import SimpleNamespace

import pytest

from grpo.judge import Judge


def test_judge_invalid_json_never_zero(monkeypatch):
    j = Judge.__new__(Judge)
    j.config = {"retries": 1}
    j.model = "test"
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"semantic":NaN}'),
                )
            ]
        )

    j.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("grpo.judge.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="bounded retries"):
        j.score("prompt", "completion", "reference")
    assert len(calls) == 2


def test_context_overflow_is_not_retried():
    from openai import BadRequestError

    from grpo.judge import JudgeContextError

    calls = []
    error = {
        "type": "exceed_context_size_error",
        "message": "request (2191 tokens) exceeds the available context size (2048 tokens)",
        "n_ctx": 2048,
    }

    def create(**kwargs):
        calls.append(kwargs)
        raise BadRequestError(
            "context overflow",
            response=SimpleNamespace(status_code=400, headers={}, request=None),
            body=error,
        )

    j = Judge.__new__(Judge)
    j.config = {"retries": 2}
    j.model = "test"
    j.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(JudgeContextError, match="not retried"):
        j.score("prompt", "completion", "reference")
    assert len(calls) == 1


def test_per_slot_context_guard(monkeypatch):
    import io

    from grpo.judge import JudgeContextError, check_context

    props = {"default_generation_settings": {"n_ctx": 2048}, "total_slots": 16}
    monkeypatch.setattr(
        "grpo.judge.urlopen", lambda *a, **k: io.StringIO(json.dumps(props))
    )
    config = {"url": "http://judge", "min_context_tokens": 16384}
    with pytest.raises(JudgeContextError, match="262144"):
        check_context(config)
    props["default_generation_settings"]["n_ctx"] = 16384
    assert check_context(config) == {"context_per_slot": 16384, "slots": 16}


def test_truncated_judgment_retries_with_bounded_larger_output_budget(monkeypatch):
    j = Judge.__new__(Judge)
    j.config = {"retries": 2, "max_tokens": 256}
    j.model = "test"
    budgets = []

    def create(**kwargs):
        budgets.append(kwargs["max_tokens"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length" if len(budgets) < 3 else "stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "semantic": 2,
                                "instruction": 3,
                                "grounding": 4,
                                "task": 3,
                                "reason": "valid assessment",
                            }
                        )
                    ),
                )
            ],
            model="test",
            id="request",
            usage=None,
        )

    j.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("grpo.judge.time.sleep", lambda _: None)
    result = j.score("prompt", "completion", "reference")
    assert budgets == [256, 512, 1024]
    assert result["reward"] == 0.75 and len(result["prior_errors"]) == 2
    assert result["max_tokens_budget"] == 1024


def test_missing_field_retry_changes_input_and_uses_strict_schema(monkeypatch):
    calls = []
    good = {'semantic': 2, 'instruction': 3, 'grounding': 4, 'task': 3, 'reason': 'valid'}
    def create(**kwargs):
        calls.append(kwargs)
        obj = dict(good)
        if len(calls) == 1:
            del obj['instruction']
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(obj)))], model='test', id='request', usage=None)
    j = Judge.__new__(Judge)
    j.config, j.model = {'retries': 1}, 'test'
    j.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr('grpo.judge.time.sleep', lambda _: None)
    result = j.score('prompt', 'completion', 'reference')
    assert result['reward'] == .75
    assert calls[0]['response_format']['type'] == 'json_schema'
    assert calls[0]['response_format']['json_schema']['strict'] is True
    assert 'instruction' in calls[0]['response_format']['json_schema']['schema']['required']
    assert calls[0]['messages'][:2] == calls[1]['messages'][:2]
    assert len(calls[1]['messages']) == 3
    assert 'instruction' in calls[1]['messages'][2]['content']
    assert len(result['prior_errors']) == 1


def test_scoring_cycle_retries_only_failed_sample_and_preserves_successes():
    from collections import Counter
    from grpo.online import score_samples
    calls = Counter()
    saved = []
    class FlakyJudge:
        def score(self, p, c, r):
            calls[c] += 1
            if c == 'bad' and calls[c] == 1:
                raise RuntimeError('transient malformed result')
            return {'reward': .75}
    samples = [dict(prompt='p', reference='r', completion=c) for c in ['good', 'bad']]
    score_samples(samples, FlakyJudge(), 2, lambda: saved.append(sum('judge' in s for s in samples)),
                  retry_delay=0, max_rounds=3)
    assert calls == {'good': 1, 'bad': 2}
    assert all(s['reward'] == .75 for s in samples)
    assert len(samples[1]['judge_failures']) == 1
    assert saved[-1] == 2


def test_fatal_judge_error_preserves_peer_success_and_does_not_retry():
    import time
    from grpo.judge import JudgeRequestError
    from grpo.online import score_samples
    calls, saved = [], []
    class FatalJudge:
        def score(self, p, c, r):
            calls.append(c)
            if c == 'bad':
                raise JudgeRequestError('server configuration')
            time.sleep(.02)
            return {'reward': .5}
    samples = [dict(prompt='p', reference='r', completion=c) for c in ['good', 'bad']]
    with pytest.raises(JudgeRequestError):
        score_samples(samples, FatalJudge(), 2, lambda: saved.append(sum('judge' in s for s in samples)), retry_delay=0)
    assert calls.count('bad') == 1 and samples[0]['reward'] == .5
    assert 'reward' not in samples[1] and saved[-1] == 1


def test_legacy_judge_retry_count_is_preserved():
    from grpo.online import failed_judge_attempts
    sample = {'judge_failures': ["Judge failed after bounded retries: KeyError: 'instruction' | KeyError: 'instruction' | KeyError: 'instruction'"]}
    assert failed_judge_attempts(sample) == 3
    sample['judge_failed_attempts'] = 5
    assert failed_judge_attempts(sample) == 5
