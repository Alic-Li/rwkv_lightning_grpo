"""Sequential stage boundaries, JSONL recovery, concurrent Judge and manifest checks."""

from collections import Counter
from threading import Barrier, Lock

import pytest

from grpo import stage1, stage2, stage3
from grpo.artifacts import read_rollouts
from grpo.io import file_hash


def setup_rollout(tmp_path, monkeypatch):
    data, vocab = tmp_path / "data", tmp_path / "vocab"
    data.write_text("fixture")
    vocab.write_text("fixture")
    rows = [
        {
            "id": str(i),
            "prompt": f"prompt {i}",
            "prompt_ids": [i + 1],
            "reference": "reference",
        }
        for i in range(4)
    ]
    monkeypatch.setattr(
        stage1, "load_policy", lambda *a: (None, {"policy_version": "v0"})
    )
    monkeypatch.setattr(stage1, "TRIE_TOKENIZER", lambda *a: None)
    monkeypatch.setattr(stage1, "prompts", lambda *a: iter(rows))
    config = {
        "sampling": {"max_new_tokens": 1},
        "group_size": 2,
        "prompts_per_batch": 1,
        "output": str(tmp_path / "raw/rollouts.jsonl"),
        "vocab": str(vocab),
        "data": str(data),
        "num_prompts": 4,
        "progress": False,
    }
    return config


def test_three_stages_resume_and_concurrent_judge(tmp_path, monkeypatch):
    c = setup_rollout(tmp_path, monkeypatch)
    seen, scored = Counter(), Counter()
    crash, failure = [True], [True]
    barrier, mutex = Barrier(2), Lock()
    active = {"n": 0, "peak": 0, "calls": 0}

    class FakeJudge:
        def __init__(self, config):
            pass

        def score(self, p, completion, reference):
            with mutex:
                active["calls"] += 1
                first_pair = active["calls"] <= 2
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
                scored[completion] += 1
            try:
                if first_pair:
                    barrier.wait(5)
                if completion == "2" and failure[0]:
                    raise RuntimeError("simulated timeout")
                return {"reward": float(completion) / 8}
            finally:
                with mutex:
                    active["n"] -= 1

    def rollout(policy, prompts, tokenizer, config, seeds, **callbacks):
        for i, seed in enumerate(seeds):
            seen[seed] += 1
            callbacks["on_complete"](
                i,
                {
                    "completion_ids": [3, 0],
                    "completion": str(sum(seen.values())),
                    "old_logprobs": [-1.0, -2.0],
                    "mask": [1, 1],
                    "seed": seed,
                },
            )
            if crash[0] and sum(seen.values()) == 3:
                raise RuntimeError("producer crash")

    monkeypatch.setattr(stage1, "rollout_batch", rollout)
    monkeypatch.setattr(stage2, "Judge", FakeJudge)
    j = {
        "rollouts": c["output"],
        "output": str(tmp_path / "graded/manifest.json"),
        "judge": {"concurrency": 2},
        "progress": False,
    }
    with pytest.raises(RuntimeError, match="producer crash"):
        stage1.run(c)
    assert active["calls"] == 0  # Rollout never calls Judge.
    with pytest.raises(ValueError, match="Stage 1 is incomplete"):
        stage2.run(j)
    assert active["calls"] == 0
    # A partially written final line is discarded only by the locked producer.
    with open(c["output"], "ab") as f:
        f.write(b'{"type":"comple')
    crash[0] = False
    stage1.run(c)
    assert len(seen) == 8 and set(seen.values()) == {1}
    assert read_rollouts(c["output"])[3]
    assert [p.name for p in (tmp_path / "raw").iterdir()] == ["rollouts.jsonl"]
    raw_hash = file_hash(c["output"])
    stage1.run(c)
    assert len(seen) == 8 and active["calls"] == 0
    with pytest.raises(RuntimeError, match="Judge failures"):
        stage2.run(j)
    assert active["peak"] == 2
    assert not (tmp_path / "graded/manifest.json").exists()
    failure[0] = False
    stage2.run(j)
    assert scored["2"] == 2 and sum(scored.values()) == 9
    assert file_hash(c["output"]) == raw_hash
    before = file_hash(j["output"])
    stage2.run({**j, "judge": {"concurrency": 1}})
    assert file_hash(j["output"]) == before and sum(scored.values()) == 9
    manifest, samples = stage3.load_dataset(j["output"], {"policy_version": "v0"})
    assert manifest["complete"] and len(samples) == 8
    for offset in range(0, len(samples), 2):
        assert sum(
            s["advantage"] for s in samples[offset : offset + 2]
        ) == pytest.approx(0)
    with pytest.raises(ValueError, match="snapshot differ"):
        stage3.load_dataset(j["output"], {"policy_version": "v1"})
    with pytest.raises(ValueError, match="policy mismatch"):
        stage1.run({**c, "group_size": 3})
    with open(c["output"], "a") as f:
        f.write(" {}\n")
    with pytest.raises(ValueError, match="changed after Stage 2"):
        stage3.load_dataset(j["output"], {"policy_version": "v0"})


def test_stages_have_no_follow_pipeline_or_cross_stage_imports():
    import ast
    from pathlib import Path

    for module, forbidden in (
        (stage1, {"judge", "stage2", "stage3"}),
        (stage2, {"model", "rollout", "stage1", "stage3"}),
        (stage3, {"judge", "rollout", "stage1", "stage2"}),
    ):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert not imports & forbidden


def test_completed_judge_requests_are_saved_even_when_peer_fails_fatally(
    tmp_path, monkeypatch
):
    import time

    from grpo.artifacts import read_judgments
    from grpo.judge import JudgeRequestError

    c = setup_rollout(tmp_path, monkeypatch)
    sequence = [0]

    def rollout(policy, prompts, tokenizer, config, seeds, **callbacks):
        for i, seed in enumerate(seeds):
            sequence[0] += 1
            callbacks["on_complete"](
                i,
                {
                    "completion_ids": [3],
                    "completion": str(sequence[0]),
                    "old_logprobs": [-1.0],
                    "mask": [1],
                },
            )

    monkeypatch.setattr(stage1, "rollout_batch", rollout)
    stage1.run(c)
    barrier = Barrier(2)
    calls = Counter()
    fatal = [True]

    class Judge:
        def __init__(self, c):
            pass

        def score(self, p, completion, reference):
            calls[completion] += 1
            if fatal[0]:
                barrier.wait(5)
                if completion == "2":
                    raise JudgeRequestError("invalid server configuration")
                time.sleep(0.05)  # Finish after the other future raised in the caller.
            return {"reward": 0.5}

    monkeypatch.setattr(stage2, "Judge", Judge)
    j = {
        "rollouts": c["output"],
        "output": str(tmp_path / "graded/manifest.json"),
        "judge": {"concurrency": 2},
        "progress": False,
    }
    with pytest.raises(JudgeRequestError):
        stage2.run(j)
    _, scores = read_judgments(tmp_path / "graded/judgments.jsonl")
    assert len(scores) == 1
    fatal[0] = False
    stage2.run(j)
    assert calls["1"] == 1 and calls["2"] == 2
    assert len(calls) == 8


def test_training_preflight_failure_does_not_claim_dataset(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    if not torch.cuda.is_available():
        pytest.skip("Stage 3 CUDA entry point requires a GPU")
    state = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    monkeypatch.setattr(
        stage3, "load_policy", lambda *a: (SimpleNamespace(initial_state=state), {})
    )
    monkeypatch.setattr(
        stage3, "load_dataset", lambda *a: ({"dataset_id": "fixture"}, [{}])
    )
    monkeypatch.setattr(
        stage3,
        "alignment",
        lambda *a: {
            "alignment/max_abs_logprob_error": 1.0,
            "alignment/mean_ratio": 1.0,
            "alignment/clip_fraction": 0.0,
        },
    )
    with pytest.raises(RuntimeError, match="alignment"):
        stage3.run(
            {
                "dataset": str(tmp_path / "manifest.json"),
                "output": str(tmp_path / "train"),
            }
        )
    assert not (tmp_path / "training-use.json").exists()
    assert not (tmp_path / "train/latest.pt").exists()
