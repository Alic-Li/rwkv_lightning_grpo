"""Online step primitives. No model loading, file-stage dependency or worker service."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic
import math

import torch
import torch.distributed as dist

from .data import advantages, collate, ppo_sum
from .judge import DIMS


def distributed():
    return dist.is_initialized()


def rank_call(fn):
    """Propagate Python failures before peers enter the next collective."""
    value, error = None, None
    try:
        value = fn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    results = [(value, error)]
    if distributed():
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, (value, error))
    errors = [e for _, e in results if e]
    if errors:
        raise RuntimeError("Online phase failed: " + " | ".join(errors))
    return [v for v, _ in results]


def root_call(fn, rank):
    return rank_call(lambda: fn() if rank == 0 else None)[0]


def score_samples(samples, judge, concurrency, save):
    """Persist each success, including successes finishing after a peer failure."""
    missing = [s for s in samples if 'judge' not in s]
    errors = []
    def score(s):
        start = monotonic()
        result = judge.score(s['prompt'], s['completion'], s['reference'])
        return result, monotonic() - start
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(score, s): s for s in missing}
        for future in as_completed(futures):
            s = futures[future]
            try:
                result, latency = future.result()
                if not math.isfinite(result['reward']):
                    raise ValueError('Nonfinite reward')
                s['judge'], s['judge_seconds'] = result, latency
                s['reward'] = result['reward']
            except Exception as exc:
                s.setdefault('judge_failures', []).append(str(exc))
                errors.append(str(exc))
            save()
    if errors:
        raise RuntimeError('Judge failures (no reward fabricated): ' + errors[0])


def group_advantages(samples, group_size, eps):
    zero = 0
    for start in range(0, len(samples), group_size):
        group = samples[start:start + group_size]
        if len(group) != group_size or len({s['prompt_id'] for s in group}) != 1:
            raise ValueError('Incomplete prompt group')
        mean, std, adv = advantages([s['reward'] for s in group], eps)
        zero += std == 0
        for s, a in zip(group, adv):
            s.update(advantage=a, group_reward_mean=mean, group_reward_std=std)
    return zero / (len(samples) / group_size)


def train_step(policy, model, optimizer, samples, c, rank=0, world=1):
    """One global token denominator and at most one optimizer step, regardless of P/G/B."""
    device = policy.initial_state.device
    micro, group = c['microbatch'], c['group_size']
    local = [s for i, s in enumerate(samples) if (i // group) % world == rank]
    tokens = sum(sum(s['mask']) for s in samples)
    if tokens <= 0:
        raise ValueError('No valid completion tokens')
    rounds = max(1, math.ceil(math.ceil(len(samples) / group / world) * group / micro))
    optimizer.zero_grad(set_to_none=True)
    before = policy.initial_state.detach().clone()
    stats = torch.zeros(6, device=device, dtype=torch.float64)
    max_error = torch.zeros((), device=device)
    skip = all(s['advantage'] == 0 for s in samples)
    for slot in range(rounds):
        batch = local[slot * micro:(slot + 1) * micro]
        if not batch:
            batch = [{**samples[0], 'mask': [0] * len(samples[0]['mask'])}]
        ids, targets, old, mask, adv = collate(batch, device)
        with model.no_sync() if world > 1 and slot < rounds - 1 else nullcontext():
            computed = []
            def forward():
                with torch.set_grad_enabled(not skip):
                    new = model(ids, targets, c.get('chunk_size', 128))
                    loss, ratio, clipped = ppo_sum(new, old, adv, mask, c.get('clip_epsilon', .2))
                    delta = torch.where(mask.bool(), new - old, torch.zeros_like(new))
                    kl = ((delta.exp() - 1 - delta) * mask).sum()
                if not torch.isfinite(loss) or not torch.isfinite(delta).all():
                    raise RuntimeError('Nonfinite forward/loss')
                computed.append((loss, ratio, clipped, delta, kl))
            rank_call(forward)
            loss, ratio, clipped, delta, kl = computed[0]
            if not skip:
                (loss * world / tokens).backward()
            stats += torch.stack([loss.detach(), ratio.detach(), clipped.detach(), kl.detach(),
                                  (delta.detach().abs() * mask).sum(),
                                  (delta.detach().square() * mask).sum()]).double()
            max_error = torch.maximum(max_error, delta.detach().abs().max())
    if distributed():
        dist.all_reduce(stats)
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    if not torch.isfinite(stats).all():
        raise RuntimeError('Nonfinite loss statistics')
    # Decode (T=1, shrinking batches) and teacher forcing (T=chunk_size) use
    # different BF16 GEMM/recurrent rounding paths. A max over thousands of
    # tokens is an outlier diagnostic, not evidence of a stale policy. Keep
    # the population ratio/clip guards below, plus the caller's strict state,
    # step and policy-version checks. Never replace the recorded sampling logp.
    max_error_exceeded = float(max_error) > c.get('alignment_tolerance', .3)
    if abs(float(stats[1] / tokens) - 1) > c.get('alignment_mean_ratio_tolerance', .02) or float(stats[2] / tokens) > c.get('alignment_max_clip_fraction', .01):
        raise RuntimeError(f'Old/new ratio alignment exceeds BF16 tolerance: ratio={float(stats[1] / tokens):.6f}, clip_fraction={float(stats[2] / tokens):.6f}')
    norm = 0.
    if not skip:
        def check_gradient():
            grad = policy.initial_state.grad
            if grad is None or not torch.isfinite(grad).all():
                raise RuntimeError('Missing/nonfinite Initial State gradient')
            if any(p.grad is not None or p.requires_grad for p in policy.backbone.parameters()):
                raise RuntimeError('Backbone must stay frozen')
        rank_call(check_gradient)
        norm = float(torch.nn.utils.clip_grad_norm_([policy.initial_state], c.get('max_grad_norm', 1.), error_if_nonfinite=True))
        rank_call(lambda: optimizer.step())
    def check_state():
        if not torch.isfinite(policy.initial_state).all():
            raise RuntimeError('Nonfinite updated state')
    rank_call(check_state)
    return {
        'training/loss': float(stats[0] / tokens), 'training/ratio_mean': float(stats[1] / tokens),
        'training/clip_fraction': float(stats[2] / tokens), 'training/approx_kl': float(stats[3] / tokens),
        'training/grad_norm': norm, 'training/state_norm': float(policy.initial_state.detach().norm()),
        'training/lr': optimizer.param_groups[0]['lr'], 'training/valid_tokens': tokens,
        'training/skipped_update': int(skip), 'policy/state_update_norm': float((policy.initial_state.detach() - before).norm()),
        'alignment/max_abs_logprob_error': float(max_error),
        'alignment/max_error_exceeded': int(max_error_exceeded),
        'alignment/mean_abs_logprob_error': float(stats[4] / tokens),
        'alignment/rms_logprob_error': float((stats[5] / tokens).sqrt()),
    }


def sample_metrics(samples):
    out = {}
    def distribution(key, values):
        v = torch.tensor(values, dtype=torch.float64)
        out.update({key + '/mean': float(v.mean()), key + '/std': float(v.std(correction=0)),
                    key + '/min': float(v.min()), key + '/max': float(v.max())})
    distribution('reward', [s['reward'] for s in samples])
    distribution('advantage', [s['advantage'] for s in samples])
    distribution('rollout/completion_length', [len(s['completion_ids']) for s in samples])
    distribution('judge/latency', [s['judge_seconds'] for s in samples])
    distribution('judge/request_latency', [latency for s in samples for latency in s['judge'].get('request_seconds', [s['judge_seconds']])])
    for dim in DIMS:
        distribution('judge/' + dim, [s['judge']['scores'][dim] for s in samples])
    failures = sum(len(s.get('judge_failures', [])) + len(s['judge'].get('prior_errors', [])) for s in samples)
    successes = len(samples)
    out.update({'judge/successes': successes, 'judge/failures': failures,
                'judge/success_rate': successes / (successes + failures),
                'judge/failure_rate': failures / (successes + failures)})
    for reason in {s['finish_reason'] for s in samples}:
        out['rollout/stop/' + reason] = sum(s['finish_reason'] == reason for s in samples) / len(samples)
    for s in samples:
        out['reward/group/' + s['prompt_id']] = s['group_reward_mean']
    return out
