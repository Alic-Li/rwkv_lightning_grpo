"""Online step primitives. No model loading, file-stage dependency or worker service."""
from contextlib import ExitStack, nullcontext
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from time import monotonic
import math

import torch
import torch.distributed as dist

from .data import advantages, collate, ppo_sum
from .judge import DIMS, JudgeRequestError


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


def failed_judge_attempts(sample):
    if 'judge_failed_attempts' in sample:
        return sample['judge_failed_attempts']
    # Legacy checkpoints recorded one exception string for several HTTP attempts.
    prefix = 'Judge failed after bounded retries: '
    return sum(len(error[len(prefix):].split(' | ')) if error.startswith(prefix) else 1
               for error in sample.get('judge_failures', []))


def score_samples(samples, judge, concurrency, save, *, rank=0, max_rounds=3,
                  retry_delay=2., on_retry=None):
    """Retry only missing scores; root persists successes, peers receive heartbeats.

    The existing request pool does all remote I/O. Main-thread collectives run
    at least once per second so DDP peers never wait through an entire Judge
    batch/retry cycle in a single collective with a finite NCCL timeout.
    """
    if max_rounds < 1 or retry_delay < 0:
        raise ValueError('Judge retry rounds/delay must be positive/nonnegative')
    def score(s):
        start = monotonic()
        result = judge.score(s['prompt'], s['completion'], s['reference'])
        return result, monotonic() - start

    for attempt in range(max_rounds):
        missing = root_call(lambda: [i for i, s in enumerate(samples) if 'judge' not in s], rank)
        if not missing:
            break
        errors, fatal = [], []
        with ExitStack() as stack:
            pending = {}
            if rank == 0:
                pool = stack.enter_context(ThreadPoolExecutor(max_workers=concurrency))
                pending = {pool.submit(score, samples[i]): i for i in missing}
            def poll():
                done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
                for future in done:
                    s = samples[pending.pop(future)]
                    if future.cancelled():
                        continue
                    try:
                        result, latency = future.result()
                        if not math.isfinite(result['reward']):
                            raise ValueError('Nonfinite reward')
                        s['judge'], s['judge_seconds'] = result, latency
                        s['reward'] = result['reward']
                    except Exception as exc:
                        prior_attempts = failed_judge_attempts(s)
                        s.setdefault('judge_failures', []).append(str(exc))
                        s['judge_failed_attempts'] = prior_attempts + len(getattr(exc, 'errors', [str(exc)]))
                        s.setdefault('judge_failed_request_seconds', []).extend(getattr(exc, 'request_seconds', []))
                        errors.append(str(exc))
                        if isinstance(exc, JudgeRequestError):
                            fatal.append(str(exc))
                            for other in pending:
                                other.cancel()
                    save()
                return {'done': not pending, 'errors': errors, 'fatal': fatal}
            while True:
                status = root_call(poll, rank)
                if status['done']:
                    break
        if status['fatal']:
            raise JudgeRequestError(status['fatal'][0])
        remaining = root_call(lambda: sum('judge' not in s for s in samples), rank)
        if remaining and attempt + 1 == max_rounds:
            raise RuntimeError(f"Judge failures (no reward fabricated): {remaining} scores missing after {max_rounds} rounds: " + status['errors'][0])
        if remaining:
            if rank == 0 and on_retry:
                on_retry(attempt + 1, remaining)
            # Backoff is small/bounded; the next poll resumes DDP heartbeats.
            import time
            time.sleep(min(retry_delay * 2**attempt, 10.))
    return root_call(lambda: samples, rank)


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


def train_step(policy, model, optimizer, samples, c, rank=0, world=1, verify_rollout=None):
    """One global token denominator and at most one optimizer step, regardless of P/G/B."""
    device = policy.initial_state.device
    micro, group = c['microbatch'], c['group_size']
    local = [s for i, s in enumerate(samples) if (i // group) % world == rank]
    if c.get('length_bucketing', True):
        local.sort(key=lambda s: len(s['prompt_ids']) + len(s['completion_ids']))
    tokens = sum(sum(s['mask']) for s in samples)
    if tokens <= 0:
        raise ValueError('No valid completion tokens')
    rounds = max(1, math.ceil(math.ceil(len(samples) / group / world) * group / micro))
    optimizer.zero_grad(set_to_none=True)
    before = policy.initial_state.detach().clone()
    stats = torch.zeros(8, device=device, dtype=torch.float64)
    max_error = torch.zeros((), device=device)
    skip = all(s['advantage'] == 0 for s in samples)
    for slot in range(rounds):
        batch = local[slot * micro:(slot + 1) * micro]
        real_positions = sum(len(s['prompt_ids']) + len(s['completion_ids']) - 1 for s in batch)
        if not batch:
            batch = [{**samples[0], 'mask': [0] * len(samples[0]['mask'])}]
        ids, targets, old, mask, adv = collate(batch, device)
        with model.no_sync() if world > 1 and slot < rounds - 1 else nullcontext():
            computed = []
            def forward():
                with torch.set_grad_enabled(not skip):
                    new = model(ids, targets, c.get('chunk_size', 128), completion_mask=mask)
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
                                  (delta.detach().square() * mask).sum(),
                                  torch.tensor(ids.numel(), device=device),
                                  torch.tensor(real_positions, device=device)]).double()
            max_error = torch.maximum(max_error, delta.detach().abs().max())
    if distributed():
        dist.all_reduce(stats)
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    if not torch.isfinite(stats).all():
        raise RuntimeError('Nonfinite loss statistics')
    # Teacher forcing and shrinking T=1 decode have distinct BF16 rounding.
    # Population limits trigger validation of the actual behavior policy,
    # rather than guessing policy freshness from a fixed numeric threshold.
    max_error_exceeded = float(max_error) > c.get('alignment_tolerance', .3)
    population_error = (abs(float(stats[1] / tokens) - 1) > c.get('alignment_mean_ratio_tolerance', .02)
                        or float(stats[2] / tokens) > c.get('alignment_max_clip_fraction', .01))
    replay_max_error = 0.
    if population_error:
        if verify_rollout is None:
            raise RuntimeError(f'Old/new ratio alignment exceeds BF16 tolerance: ratio={float(stats[1] / tokens):.6f}, clip_fraction={float(stats[2] / tokens):.6f}; behavior replay required')
        replay_max_error = max(rank_call(verify_rollout))
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
        'training/teacher_forcing_tokens': int(stats[6]),
        'training/padding_fraction': float(1 - stats[7] / stats[6]),
        'training/skipped_update': int(skip), 'policy/state_update_norm': float((policy.initial_state.detach() - before).norm()),
        'alignment/behavior_replayed': int(population_error),
        'alignment/replay_max_abs_logprob_error': replay_max_error,
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
    distribution('judge/request_latency', [latency for s in samples for latency in s['judge'].get('request_seconds', [s['judge_seconds']]) + s.get('judge_failed_request_seconds', [])])
    for dim in DIMS:
        distribution('judge/' + dim, [s['judge']['scores'][dim] for s in samples])
    failures = sum(failed_judge_attempts(s) + len(s['judge'].get('prior_errors', [])) for s in samples)
    successes = len(samples)
    out.update({'judge/successes': successes, 'judge/failures': failures,
                'judge/success_rate': successes / (successes + failures),
                'judge/failure_rate': failures / (successes + failures)})
    for reason in {s['finish_reason'] for s in samples}:
        out['rollout/stop/' + reason] = sum(s['finish_reason'] == reason for s in samples) / len(samples)
    for s in samples:
        out['reward/group/' + s['prompt_id']] = s['group_reward_mean']
    return out
