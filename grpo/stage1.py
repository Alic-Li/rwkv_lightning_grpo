"""Stage 1: GPU-only batched rollout to one resumable JSONL. No Judge imports."""

from pathlib import Path

from tqdm.auto import tqdm

from .artifacts import append_jsonl, read_rollouts
from .data import prompts
from .io import config_cli, digest, file_hash, load_policy, lock
from .rollout import rollout_batch
from .tokenizer import TRIE_TOKENIZER


def run(c):
    sampling = c["sampling"]
    if sampling.get("temperature", 1) != 1 or sampling.get("top_p", 1) != 1:
        raise ValueError("Use temperature=1 and top_p=1")
    if c["group_size"] < 2 or c["prompts_per_batch"] < 1 or c["num_prompts"] < 1:
        raise ValueError("Require G>=2, P>=1 and num_prompts>=1")
    if sampling["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    path = Path(c["output"])
    if path.suffix != ".jsonl":
        raise ValueError("Stage 1 output must be a .jsonl file")
    with lock(path):
        policy, snapshot = load_policy(c, "cuda")
        tokenizer = TRIE_TOKENIZER(c["vocab"])
        selected = []
        for p in prompts(c["data"], tokenizer, c.get("max_prompt_tokens", 8192)):
            selected.append(p)
            if len(selected) == c["num_prompts"]:
                break
        if not selected:
            raise ValueError("No eligible prompts")
        spec = {
            "format_version": 2,
            "snapshot": snapshot,
            "data_sha256": file_hash(c["data"]),
            "vocab_sha256": file_hash(c["vocab"]),
            "sampling": sampling,
            "group_size": c["group_size"],
            "prompt_ids": [p["id"] for p in selected],
            "seed": c.get("seed", 42),
        }
        saved_spec, saved_prompts, saved_samples, complete = read_rollouts(
            path, repair_tail=True, keep_samples=False
        )
        if saved_spec is not None and saved_spec != spec:
            raise ValueError("Rollout configuration/policy mismatch; use a new JSONL")
        if complete:
            print("Rollout JSONL already complete:", path)
            return
        total = len(selected) * c["group_size"]
        with (
            path.open("a", encoding="utf-8") as stream,
            tqdm(
                total=total,
                initial=len(saved_samples),
                desc="Rollout",
                unit="traj",
                dynamic_ncols=True,
                mininterval=0.5,
                miniters=0,
                disable=not c.get("progress", True),
            ) as progress,
        ):
            if saved_spec is None:
                append_jsonl(stream, {"type": "metadata", "spec": spec})
            for p in selected:
                if p["id"] in saved_prompts:
                    if saved_prompts[p["id"]] != p:
                        raise ValueError("Saved prompt changed")
                else:
                    append_jsonl(stream, {"type": "prompt", "prompt": p})

            def on_progress(phase, active, position):
                progress.set_postfix(
                    phase=phase, active=active, token=position, refresh=False
                )
                progress.update(0)

            for start in range(0, len(selected), c["prompts_per_batch"]):
                jobs = []
                for p in selected[start : start + c["prompts_per_batch"]]:
                    for g in range(c["group_size"]):
                        if (p["id"], g) not in saved_samples:
                            seed = int(digest([spec["seed"], p["id"], g])[:15], 16)
                            jobs.append((p, g, seed))
                if not jobs:
                    continue

                def on_complete(i, sample, batch_jobs=jobs):
                    p, g, seed = batch_jobs[i]
                    append_jsonl(
                        stream,
                        {
                            "type": "completion",
                            "prompt_id": p["id"],
                            "sample_index": g,
                            "sample": sample,
                        },
                    )
                    progress.update(1)

                rollout_batch(
                    policy,
                    [p["prompt_ids"] for p, g, seed in jobs],
                    tokenizer,
                    sampling,
                    [seed for p, g, seed in jobs],
                    on_complete=on_complete,
                    on_progress=on_progress,
                )
            append_jsonl(stream, {"type": "complete"})
        print("Rollout JSONL complete:", path)


if __name__ == "__main__":
    run(config_cli(__doc__))
