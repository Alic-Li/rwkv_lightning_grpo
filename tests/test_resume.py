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
