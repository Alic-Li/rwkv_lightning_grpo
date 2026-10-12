"""Stage 2: concurrent Judge of a COMPLETE rollout JSONL; export manifest.json."""

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

from tqdm.auto import tqdm

from .artifacts import append_jsonl, checked_manifest, read_judgments, read_rollouts
from .data import advantages
from .io import atomic_json, config_cli, digest, file_hash, lock
from .judge import RUBRIC, Judge, JudgeRequestError


def run(c):
    source = Path(c["rollouts"]).resolve()
    output = Path(c["output"]).resolve()
    if not source.is_file():
        raise ValueError("Run Stage 1 first: rollout JSONL does not exist")
    if output.name != "manifest.json":
        raise ValueError("Stage 2 output must end in manifest.json")
    concurrency = c["judge"].get("concurrency", 8)
    if not 1 <= concurrency <= 16:
        raise ValueError("Judge concurrency must be 1..16")
    journal = output.parent / "judgments.jsonl"
    # Separate stages: refuse simultaneous access to an unfinished producer file.
    with lock(source), lock(journal):
        rollout_spec, prompts, samples, complete = read_rollouts(source)
        if not complete:
            raise ValueError(
                "Stage 1 is incomplete; finish Rollout before starting Stage 2"
            )
        runtime = {
            "api_key",
            "concurrency",
            "timeout",
            "retries",
            "min_context_tokens",
            "max_retry_tokens",
        }
        score_spec = {
            "rollout_sha256": file_hash(source),
            "rubric": RUBRIC,
            "judge": {k: v for k, v in c["judge"].items() if k not in runtime},
            "advantage_eps": c.get("advantage_eps", 1e-6),
        }
        if output.exists():
            manifest = checked_manifest(output)
            if manifest["scoring"] != score_spec:
                raise ValueError(
                    "Judge configuration or Rollout changed; use a new output directory"
                )
            print("Judge manifest already complete:", output)
            return
        if (output.parent / "training-use.json").exists():
            raise ValueError(
                "Dataset already used by Stage 3; cannot regenerate its manifest"
            )
        saved_spec, scores = read_judgments(journal, repair_tail=True)
        if saved_spec is not None and saved_spec != score_spec:
            raise ValueError("Judge resume configuration/Rollout mismatch")
        if any(key not in samples for key in scores):
            raise ValueError("Judgment refers to an unknown trajectory")
        missing = iter(key for key in samples if key not in scores)
        failed = []
        judge = Judge(c["judge"]) if len(scores) != len(samples) else None
        with (
            journal.open("a", encoding="utf-8") as stream,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
            tqdm(
                total=len(samples),
                initial=len(scores),
                desc="Judge",
                unit="traj",
                dynamic_ncols=True,
                mininterval=0.5,
                miniters=0,
                disable=not c.get("progress", True),
            ) as progress,
        ):
            if saved_spec is None:
                append_jsonl(stream, {"type": "metadata", "spec": score_spec})
            pending = {}
            journal_lock = Lock()

            def score_and_save(key):
                pid, index = key
                p = prompts[pid]
                try:
                    result = judge.score(
                        p["prompt"], samples[key]["completion"], p["reference"]
                    )
                except Exception as exc:
                    with journal_lock:
                        append_jsonl(
                            stream,
                            {
                                "type": "error",
                                "prompt_id": pid,
                                "sample_index": index,
                                "error": str(exc),
                            },
                        )
                    raise
                with journal_lock:
                    append_jsonl(
                        stream,
                        {
                            "type": "judgment",
                            "prompt_id": pid,
                            "sample_index": index,
                            "judge": result,
                        },
                    )
                return result

            def fill_slots():
                while len(pending) < concurrency:
                    key = next(missing, None)
                    if key is None:
                        break
                    future = pool.submit(score_and_save, key)
                    pending[future] = key

            fill_slots()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pid, index = pending.pop(future)
                    try:
                        result = future.result()
                    except JudgeRequestError:
                        for other in pending:
                            other.cancel()
                        raise
                    except Exception as exc:
                        failed.append((pid, index, str(exc)))
                    else:
                        scores[pid, index] = result
                        progress.update(1)
                fill_slots()
                progress.set_postfix(
                    in_flight=len(pending), failed=len(failed), refresh=False
                )
                progress.update(0)
        if failed:
            raise RuntimeError(
                f"{len(failed)} Judge failures; rerun Stage 2 to retry only missing scores: {failed[0]}"
            )
        groups = []
        for pid in rollout_spec["prompt_ids"]:
            rewards = [
                scores[pid, g]["reward"] for g in range(rollout_spec["group_size"])
            ]
            mean, std, adv = advantages(rewards, score_spec["advantage_eps"])
            groups.append(
                {
                    "prompt_id": pid,
                    "rewards": rewards,
                    "advantages": adv,
                    "reward_mean": mean,
                    "reward_std": std,
                }
            )
        manifest = {
            "format_version": 2,
            "complete": True,
            "spec": rollout_spec,
            "scoring": score_spec,
            "groups": groups,
            "rollouts": {
                "path": os.path.relpath(source, output.parent),
                "sha256": score_spec["rollout_sha256"],
            },
            "judgments": {"path": journal.name, "sha256": file_hash(journal)},
        }
        manifest["dataset_id"] = digest(manifest)
        atomic_json(output, manifest)
        print("Judge manifest complete:", output)


if __name__ == "__main__":
    run(config_cli(__doc__))
