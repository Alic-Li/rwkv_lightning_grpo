"""Stage 3: offline state-only GRPO from the Stage 2 manifest; native DDP."""

import os
import random
from contextlib import ExitStack, nullcontext
from pathlib import Path
from time import monotonic

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from .artifacts import checked_manifest, read_judgments, read_rollouts
from .data import advantages, collate, ppo_sum
from .io import (
    atomic_json,
    atomic_torch,
    config_cli,
    digest,
    load_policy,
    lock,
    read_json,
    state_hash,
)
from .training import TrainingPolicy


def load_dataset(path, snapshot):
    path = Path(path)
    m = checked_manifest(path)
    if m["spec"]["snapshot"] != snapshot:
        raise ValueError(
            "Dataset and initial policy snapshot differ; regenerate rollouts"
        )
    spec, prompts, raw, complete = read_rollouts(path.parent / m["rollouts"]["path"])
    scoring, scores = read_judgments(path.parent / m["judgments"]["path"])
    if not complete or spec != m["spec"] or scoring != m["scoring"]:
        raise ValueError("JSONL metadata does not match manifest")
    if set(raw) != set(scores):
        raise ValueError("Incomplete judgments")
    if [g["prompt_id"] for g in m["groups"]] != spec["prompt_ids"]:
        raise ValueError("Invalid manifest group ordering")
    samples = []
    for group in m["groups"]:
        pid = group["prompt_id"]
        rewards = [scores[pid, g]["reward"] for g in range(spec["group_size"])]
        mean, std, advs = advantages(rewards, scoring["advantage_eps"])
        if (
            rewards != group["rewards"]
            or len(advs) != len(group["advantages"])
            or abs(mean - group["reward_mean"]) > 1e-6
            or abs(std - group["reward_std"]) > 1e-6
            or any(abs(a - b) > 1e-6 for a, b in zip(advs, group["advantages"]))
        ):
            raise ValueError("Corrupt group advantage/reward")
        for g, advantage in enumerate(advs):
            samples.append(
                {
                    **raw[pid, g],
                    "prompt_ids": prompts[pid]["prompt_ids"],
                    "reward": rewards[g],
                    "advantage": advantage,
                }
            )
    return m, samples


@torch.no_grad()
def alignment(policy, samples, chunk_size, clip_epsilon=0.2, microbatch=1):
    maximum = 0.0
    total = 0.0
    count = 0
    clipped = 0
    for start in range(0, len(samples), microbatch):
        ids, targets, old, mask, _ = collate(
            samples[start : start + microbatch], policy.initial_state.device
        )
        new = policy(ids, targets, chunk_size)
        delta = (new - old)[mask.bool()]
        if not torch.isfinite(delta).all():
            raise RuntimeError("Nonfinite old/new logprob alignment")
        maximum = max(maximum, float(delta.abs().max()))
        ratio = delta.exp()
        total += float(ratio.sum())
        clipped += int(((ratio - 1).abs() > clip_epsilon).sum())
        count += delta.numel()
    return {
        "alignment/max_abs_logprob_error": maximum,
        "alignment/mean_ratio": total / count,
        "alignment/clip_fraction": clipped / count,
    }


def run(c):
    if c.get("epochs", 1) != 1 or c.get("kl_beta", 0) != 0:
        raise ValueError(
            "v1 requires K=1 and KL beta=0; collect fresh rollouts for the next iteration"
        )
    if (
        c.get("alignment_samples", c.get("microbatch", 16)) < 1
        or c.get("learning_rate", 1e-4) <= 0
        or c.get("max_grad_norm", 1.0) <= 0
    ):
        raise ValueError(
            "Alignment count, learning rate and max gradient norm must be positive"
        )
    if not 0 < c.get("clip_epsilon", 0.2) < 1:
        raise ValueError("clip_epsilon must be between 0 and 1")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if (
        int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > torch.cuda.device_count()
        or local >= torch.cuda.device_count()
    ):
        raise RuntimeError("DDP requires one distinct CUDA device per local rank")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        with ExitStack() as stack:
            root = Path(c["output"])
            root.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                stack.enter_context(lock(root / ".training.lock"))
            policy, snapshot = load_policy(c, device)
            manifest, samples = load_dataset(c["dataset"], snapshot)
            random.Random(c.get("seed", 42)).shuffle(samples)
            micro = c.get("microbatch", 16)
            accum = c.get("gradient_accumulation", 1)
            if micro < 1 or accum < 1 or c.get("chunk_size", 128) < 1:
                raise ValueError("Batch/chunk sizes must be positive")
            train_spec = {
                **c,
                "world_size": world,
                "microbatch": micro,
                "gradient_accumulation": accum,
                "activation_checkpointing": c.get("activation_checkpointing", True),
            }
            train_spec.pop("resume", None)
            train_spec.pop("stop_after_steps", None)
            train_id = digest([manifest["dataset_id"], str(root.resolve()), train_spec])
            ledger = Path(c["dataset"]).parent / "training-use.json"
            if rank == 0:
                stack.enter_context(
                    lock(Path(c["dataset"]).parent / ".training-use.lock")
                )
                if ledger.exists():
                    used = read_json(ledger)
                    if used["train_id"] != train_id:
                        raise ValueError(
                            "Dataset is already assigned to another training run; generate fresh rollouts"
                        )
                    if used["complete"] and not (root / "latest.pt").exists():
                        raise ValueError(
                            "Consumed dataset has no checkpoint; refusing to replay it"
                        )
                    if not c.get("resume"):
                        raise ValueError(
                            "Existing run: use resume=true; dataset cannot be silently reused"
                        )
            optimizer = torch.optim.AdamW(
                [policy.initial_state], lr=c.get("learning_rate", 1e-4), weight_decay=0
            )
            cursor = step = 0
            checkpoint = root / "latest.pt"
            if c.get("resume") and checkpoint.exists():
                ck = torch.load(checkpoint, map_location=device, weights_only=True)
                if ck["train_id"] != train_id or ck["base_snapshot"] != snapshot:
                    raise ValueError("Training resume metadata mismatch")
                if ck["state_version"] != state_hash(ck["initial_state"]):
                    raise ValueError("Checkpoint state fingerprint mismatch")
                policy.initial_state.data.copy_(ck["initial_state"])
                optimizer.load_state_dict(ck["optimizer"])
                cursor, step = ck["cursor"], ck["step"]
            if world > 1:
                dist.barrier()
            if cursor == len(samples):
                if rank == 0:
                    print("Training already complete:", checkpoint, flush=True)
                return
            if cursor == 0:
                metrics = alignment(
                    policy,
                    samples[: c.get("alignment_samples", micro)],
                    c.get("chunk_size", 128),
                    c.get("clip_epsilon", 0.2),
                    micro,
                )
                if rank == 0:
                    print(metrics, flush=True)
                    atomic_json(root / "alignment.json", metrics)
                if (
                    metrics["alignment/max_abs_logprob_error"]
                    > c.get("alignment_tolerance", 0.15)
                    or abs(metrics["alignment/mean_ratio"] - 1)
                    > c.get("alignment_mean_ratio_tolerance", 0.02)
                    or metrics["alignment/clip_fraction"]
                    > c.get("alignment_max_clip_fraction", 0.01)
                ):
                    raise RuntimeError(
                        "Old/new logprob alignment exceeds BF16 tolerance"
                    )
            # Claim the dataset only after preflight succeeds. The dataset lock
            # is already held, so another trainer cannot race this registration.
            if rank == 0 and not ledger.exists():
                atomic_json(
                    ledger,
                    {
                        "train_id": train_id,
                        "output": str(root.resolve()),
                        "complete": False,
                    },
                )
            training_policy = TrainingPolicy(
                policy, activation_checkpointing=c.get("activation_checkpointing", True)
            )
            model = (
                DDP(training_policy, device_ids=[local], broadcast_buffers=False)
                if world > 1
                else training_policy
            )
            logger = None
            if rank == 0:
                import swanlab

                logger = swanlab
                logger.init(
                    project=c.get("swanlab_project", "rwkv-state-grpo"),
                    name=root.name,
                    mode=c.get("swanlab_mode", "offline"),
                    log_dir=str(root / "swanlog"),
                    config=train_spec,
                )
            global_batch = micro * accum * world
            progress = stack.enter_context(
                tqdm(
                    total=len(samples),
                    initial=cursor,
                    desc="Train",
                    unit="traj",
                    dynamic_ncols=True,
                    mininterval=0.5,
                    miniters=0,
                    disable=rank != 0 or not c.get("progress", True),
                )
            )
            while cursor < len(samples):
                step_started = monotonic()
                window = samples[cursor : cursor + global_batch]
                token_count = sum(sum(s["mask"]) for s in window)
                optimizer.zero_grad(set_to_none=True)
                before = policy.initial_state.detach().clone()
                stats = torch.zeros(3, device=device, dtype=torch.float64)
                rounds = (len(window) + micro * world - 1) // (micro * world)
                for a in range(rounds):
                    offset = (a * world + rank) * micro
                    local_samples = window[offset : offset + micro]
                    if not local_samples:
                        # All ranks execute the same number of forwards/backwards; dummy has no loss.
                        dummy = {**window[0], "mask": [0] * len(window[0]["mask"])}
                        local_samples = [dummy]
                    ids, targets, old, mask, adv = collate(local_samples, device)
                    sync = (
                        model.no_sync()
                        if world > 1 and a < rounds - 1
                        else nullcontext()
                    )
                    with sync:
                        new = model(ids, targets, c.get("chunk_size", 128))
                        loss, ratio, clipped = ppo_sum(
                            new, old, adv, mask, c.get("clip_epsilon", 0.2)
                        )
                        # DDP averages gradients over ranks. Normalize once over ALL valid
                        # completion tokens across ranks AND accumulation microbatches.
                        (loss * (world / token_count)).backward()
                    stats += torch.stack(
                        (loss.detach(), ratio.detach(), clipped.detach())
                    ).double()
                    if rank == 0:
                        progress.set_postfix(
                            step=step + 1,
                            micro=f"{a + 1}/{rounds}",
                            tokens=token_count,
                            refresh=False,
                        )
                        progress.update(
                            min(micro * world, max(0, len(window) - a * world * micro))
                        )
                grad = policy.initial_state.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise RuntimeError("Missing/nonfinite Initial State gradient")
                if any(p.grad is not None for p in policy.backbone.parameters()):
                    raise RuntimeError("Backbone received gradients")
                norm = torch.nn.utils.clip_grad_norm_(
                    [policy.initial_state],
                    c.get("max_grad_norm", 1.0),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                if not torch.isfinite(policy.initial_state).all():
                    raise RuntimeError("Nonfinite state after optimizer step")
                if world > 1:
                    dist.all_reduce(stats)
                cursor += len(window)
                step += 1
                if rank == 0:
                    metrics = {
                        "loss": float(stats[0] / token_count),
                        "ratio": float(stats[1] / token_count),
                        "clip_fraction": float(stats[2] / token_count),
                        "gradient_norm": float(norm),
                        "state_update_norm": float(
                            (policy.initial_state.detach() - before).norm()
                        ),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "valid_tokens": token_count,
                        "reward": sum(s["reward"] for s in window) / len(window),
                        "advantage": sum(s["advantage"] for s in window) / len(window),
                        "advantage_abs": sum(abs(s["advantage"]) for s in window)
                        / len(window),
                        "trajectories_seen": cursor,
                        "step_seconds": monotonic() - step_started,
                        "microbatch": micro,
                    }
                    logger.log(metrics, step=step)
                    with open(root / "metrics.jsonl", "a") as f:
                        import json

                        f.write(json.dumps({"step": step, **metrics}) + "\n")
                    progress.set_postfix(
                        step=step,
                        loss=f"{metrics['loss']:.4f}",
                        ratio=f"{metrics['ratio']:.4f}",
                        reward=f"{metrics['reward']:.4f}",
                        refresh=False,
                    )
                    state = policy.initial_state.detach().cpu()
                    atomic_torch(
                        checkpoint,
                        {
                            "format_version": 1,
                            "initial_state": state,
                            "state_version": state_hash(state),
                            "backbone_sha256": snapshot["backbone_sha256"],
                            "base_snapshot": snapshot,
                            "dataset_id": manifest["dataset_id"],
                            "train_id": train_id,
                            "optimizer": optimizer.state_dict(),
                            "cursor": cursor,
                            "step": step,
                            "complete": cursor == len(samples),
                            "layout": "layer,head,value,key",
                        },
                    )
                    atomic_json(
                        ledger,
                        {
                            "train_id": train_id,
                            "output": str(root.resolve()),
                            "complete": cursor == len(samples),
                        },
                    )
                if world > 1:
                    dist.barrier()
                if c.get("stop_after_steps") and step >= c["stop_after_steps"]:
                    break
            progress.close()
            if logger:
                logger.finish()
            if rank == 0:
                print(
                    "Checkpoint:",
                    checkpoint,
                    "complete:",
                    cursor == len(samples),
                    flush=True,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(config_cli(__doc__))
