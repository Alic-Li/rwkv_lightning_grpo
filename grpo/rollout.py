"""Batched RWKV rollout with independent states and per-trajectory RNG streams."""

import random

import torch


def decode_generated(tokenizer, tokens):
    return b"".join(
        tokenizer.idx2token.get(t, f"<|token:{t}|>".encode()) for t in tokens
    ).decode("utf-8", errors="replace")


@torch.no_grad()
def prefill_batch(policy, prompts, chunk_size, on_progress=None):
    """Prefill unique prompts together without ever advancing a row through padding.

    Keep fixed chunk boundaries and run short final chunks separately. Returned
    states include the final REAL token, in the original prompt order.
    """
    if chunk_size < 1 or not prompts or any(not p for p in prompts):
        raise ValueError("Prefill requires nonempty prompts and positive chunk size")
    device = policy.initial_state.device
    active = list(range(len(prompts)))
    states, shifts = policy.start(len(active)), None
    result = [None] * len(prompts)
    # Preserve the training chunk grid (0, chunk_size, ...), even when a
    # shorter row ends mid-chunk. Short tails run separately so neither padding
    # nor a neighbour's length changes another prompt's recurrent boundaries.
    for pos in range(0, max(map(len, prompts)), chunk_size):
        widths = {}
        for row, i in enumerate(active):
            width = min(chunk_size, len(prompts[i]) - pos)
            widths.setdefault(width, []).append(row)
        next_active, next_states, next_shifts = [], None, None
        for width, rows in widths.items():
            index = torch.tensor(rows, dtype=torch.long, device=device)
            group = [active[row] for row in rows]
            ids = torch.tensor(
                [prompts[i][pos : pos + width] for i in group], device=device
            )
            logits, out_states, out_shifts = policy.backbone(
                ids,
                states.index_select(1, index),
                None if shifts is None else shifts.index_select(2, index),
                last_only=True,
            )
            keep = []
            for row, i in enumerate(group):
                if pos + width == len(prompts[i]):
                    result[i] = (
                        logits[row, -1].clone(),
                        out_states[:, row].clone(),
                        out_shifts[:, :, row].clone(),
                    )
                else:
                    keep.append(row)
            if keep:
                # Only the full-width group can contain unfinished prompts.
                index = torch.tensor(keep, dtype=torch.long, device=device)
                next_states = out_states.index_select(1, index)
                next_shifts = out_shifts.index_select(2, index)
                next_active = [group[row] for row in keep]
        if on_progress:
            on_progress(
                phase="prefill",
                active=len(active),
                position=min(pos + chunk_size, max(map(len, prompts))),
            )
        active, states, shifts = next_active, next_states, next_shifts
    return result


@torch.no_grad()
def rollout_batch(
    policy,
    prompt_ids,
    tokenizer,
    config,
    seeds,
    on_complete=None,
    on_progress=None,
):
    """One GPU decode forward for ALL live trajectories at each generation step.

    Prompt prefill is shared only for identical token sequences. torch.stack
    allocates separate FP32 WKV and BF16 shift storage for every completion.
    """
    if len(prompt_ids) != len(seeds):
        raise ValueError("One seed is required per trajectory")
    if not prompt_ids:
        return []
    if config["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    if config.get("temperature", 1) != 1 or config.get("top_p", 1) != 1:
        raise ValueError("Rollout requires temperature=1 and top_p=1")
    device = policy.initial_state.device
    unique, indices, lookup = [], [], {}
    for prompt in prompt_ids:
        key = tuple(prompt)
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(prompt)
        indices.append(lookup[key])
    prefilled = prefill_batch(
        policy, unique, config.get("prefill_chunk", 128), on_progress
    )
    logits = torch.stack([prefilled[i][0] for i in indices])
    states = torch.stack([prefilled[i][1] for i in indices], dim=1)
    shifts = torch.stack([prefilled[i][2] for i in indices], dim=2)
    del prefilled
    total = len(seeds)
    rng = [random.Random(seed) for seed in seeds]
    tokens, old = [[] for _ in seeds], [[] for _ in seeds]
    results = [None] * total
    active = list(range(total))
    eos = config.get("eos_id", 0)
    stops = config.get("stop_strings", ["\n\nUser:"])
    for step in range(config["max_new_tokens"]):
        logp = logits.float().log_softmax(-1)
        # Independent uniform draws; the categorical inverse CDF runs on the GPU
        # for the entire batch. No per-row CUDA multinomial/synchronization loop.
        uniforms = torch.tensor(
            [rng[i].random() for i in active], device=device, dtype=torch.float32
        )[:, None]
        cdf = logp.exp().cumsum(-1)
        cdf.div_(cdf[:, -1:].clone())
        uniforms.clamp_max_(1.0 - torch.finfo(torch.float32).eps)
        sampled = torch.searchsorted(cdf, uniforms.contiguous(), right=True).clamp_max(
            logp.shape[-1] - 1
        )
        sampled_logp = logp.gather(1, sampled)
        sampled_ids = sampled[:, 0].tolist()
        sampled_old = sampled_logp[:, 0].tolist()
        keep = []
        for row, i in enumerate(active):
            token = sampled_ids[row]
            tokens[i].append(token)
            old[i].append(sampled_old[row])
            finish = None
            if token == eos:
                finish = "eos"
            elif any(s in decode_generated(tokenizer, tokens[i]) for s in stops):
                finish = "stop"
            elif step + 1 == config["max_new_tokens"]:
                finish = "length"
            if finish is None:
                keep.append(row)
                continue
            sample = {
                "completion_ids": tokens[i],
                "completion": decode_generated(
                    tokenizer, [t for t in tokens[i] if t != eos]
                ),
                "old_logprobs": old[i],
                "mask": [1] * len(tokens[i]),
                "finish_reason": finish,
                "seed": seeds[i],
                "rollout_batch_size": total,
                "sampling_rng": "python_random_inverse_cdf_v1",
            }
            results[i] = sample
            if on_complete:
                on_complete(i, sample)
        if on_progress:
            on_progress(phase="decode", active=len(keep), position=step + 1)
        if not keep:
            break
        index = torch.tensor(keep, dtype=torch.long, device=device)
        if len(keep) != len(active):
            states = states.index_select(1, index)
            shifts = shifts.index_select(2, index)
        inputs = sampled.index_select(0, index)
        active = [active[row] for row in keep]
        logits, states, shifts = policy.backbone(inputs, states, shifts, last_only=True)
        logits = logits[:, -1]
    return results


def rollout(policy, prompt_ids, tokenizer, config, seed):
    return rollout_batch(policy, [prompt_ids], tokenizer, config, [seed])[0]
