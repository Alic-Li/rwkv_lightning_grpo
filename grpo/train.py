"""Single-entry, recoverable Online State-only GRPO (native PyTorch DDP)."""
import json
import os
import random
import uuid
from datetime import timedelta
from contextlib import ExitStack
from pathlib import Path
from time import monotonic

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from .data import prompts
from .io import atomic_json, atomic_torch, config_cli, digest, file_hash, load_policy, lock, read_json, state_hash
from .judge import Judge
from .online import distributed, group_advantages, rank_call, root_call, sample_metrics, score_samples, train_step
from .rollout import rollout_batch
from .tokenizer import TRIE_TOKENIZER
from .training import TrainingPolicy


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def atomic_jsonl(path, records):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


SWANLAB_METRICS = {
    'training/loss', 'training/lr', 'training/grad_norm', 'training/state_norm',
    'training/ratio_mean', 'training/clip_fraction', 'training/approx_kl',
    'training/valid_tokens', 'training/skipped_update',
    'reward/mean', 'reward/std', 'reward/zero_variance_group_fraction',
    'rollout/completion_length/mean', 'judge/success_rate',
    'policy/epoch', 'policy/version', 'policy/state_update_norm',
}


def log_step(logger, root, step, metrics):
    """Upload only the core scalar dashboard; full records stay in local JSONL."""
    logger.log({key: value for key, value in metrics.items() if key in SWANLAB_METRICS}, step=step)
    atomic_json(root / 'logged.json', {'step': step})


def validate(c):
    # Retain the legacy trace_text_bytes default for checkpoint identity only.
    for key, default in [('prompts_per_step', 4), ('group_size', 8), ('microbatch', 16),
                         ('epochs', 1), ('checkpoint_interval', 1), ('chunk_size', 128), ('trace_text_bytes', 32768)]:
        c.setdefault(key, default)
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['group_size'] < 2 or c.get('max_steps', 0) < 0:
        raise ValueError('Require group_size >= 2 and max_steps >= 0')
    c.setdefault('judge_concurrency', c['judge'].get('concurrency', 8))
    if not 1 <= c['judge_concurrency'] <= 16:
        raise ValueError('Judge concurrency must be 1..16')
    if c.get('learning_rate', 1e-4) <= 0 or c.get('max_grad_norm', 1.) <= 0 or not 0 < c.get('clip_epsilon', .2) < 1:
        raise ValueError('Invalid learning rate, gradient norm or PPO clip')
    if c.get('kl_beta', 0) != 0:
        raise ValueError('This state-only objective uses kl_beta=0')
    if c['sampling'].get('prefill_chunk', 128) != c['chunk_size']:
        raise ValueError('Rollout prefill_chunk must match training chunk_size')
    if c['sampling']['max_new_tokens'] < 1 or c['sampling'].get('temperature', 1) != 1 or c['sampling'].get('top_p', 1) != 1:
        raise ValueError('Require positive generation length, temperature=1, top_p=1')


def online_loop(c, policy, snapshot, selected, tokenizer, judge, logger=None):
    """Dependencies are injected for small-model/CPU tests; production loads one backbone."""
    rank = dist.get_rank() if distributed() else 0
    world = dist.get_world_size() if distributed() else 1
    device = policy.initial_state.device
    root = Path(c['output'])
    root_call(lambda: (root / 'steps').mkdir(parents=True, exist_ok=True), rank)
    checkpoint = root / 'latest.pt'
    # Runtime stop knobs and secrets don't change the policy/data contract.
    spec = {k: v for k, v in c.items() if k not in {'resume', 'stop_after_steps', 'swanlab_mode', 'judge_retry_rounds', 'judge_retry_delay', 'length_bucketing'}}
    spec['judge'] = {k: v for k, v in c['judge'].items() if k != 'api_key'}
    train_id = digest([spec, snapshot, selected, world])
    optimizer = torch.optim.AdamW([policy.initial_state], lr=c.get('learning_rate', 1e-4), weight_decay=0)
    ck = root_call(lambda: torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists() else None, rank)
    if ck is not None and not c.get('resume'):
        raise ValueError('Existing run requires resume=true')
    if ck is None and c.get('resume'):
        raise ValueError('Resume requested but latest.pt is missing')
    if ck is not None:
        if ck['train_id'] != train_id or ck['state_version'] != state_hash(ck['initial_state']):
            raise ValueError('Checkpoint configuration/state mismatch')
        with torch.no_grad():
            policy.initial_state.copy_(ck['initial_state'].to(device))
        optimizer.load_state_dict(ck['optimizer'])
        restore_rng(ck['rng'][rank])
        if rank == 0:
            print(f"Resumed Initial WKV State: step={ck['global_step']}, policy_version={ck['policy_version']}, cursor={ck['cursor']}", flush=True)
    else:
        ck = root_call(lambda: {'global_step': 0, 'epoch': 0, 'cursor': 0, 'policy_version': 0, 'run_id': uuid.uuid4().hex}, rank)
    training = TrainingPolicy(policy, c.get('activation_checkpointing', True))
    model = DDP(training, device_ids=[device.index] if device.type == 'cuda' else None, broadcast_buffers=False) if world > 1 else training

    def commit(metrics=None):
        states = rank_call(rng_state)
        state = policy.initial_state.detach().cpu()
        ck.update(format_version=3, train_id=train_id, initial_state=state, state_version=state_hash(state),
                  backbone_sha256=snapshot['backbone_sha256'], base_snapshot=snapshot, optimizer=optimizer.state_dict(),
                  rng=states, scheduler=None, layout='layer,head,value,key', metrics=metrics)
        root_call(lambda: atomic_torch(checkpoint, ck), rank)
        if ck['global_step'] and ck['global_step'] % c['checkpoint_interval'] == 0:
            root_call(lambda: atomic_torch(root / f"state-{ck['global_step']:08d}.pt", ck), rank)

    if not checkpoint.exists():
        commit()
    def init_logger():
        if logger is None:
            return
        public_config = {**c, 'judge': spec['judge']}
        logger.init(project=c.get('swanlab_project', 'rwkv-state-grpo'), name=root.name,
                    id=ck['run_id'], resume='allow', mode=c.get('swanlab_mode', 'online'),
                    log_dir=str(root / 'swanlog'), config=public_config)
    root_call(init_logger, rank)
    def report():
        if rank == 0 and logger is not None and ck.get('metrics'):
            logged = read_json(root / 'logged.json')['step'] if (root / 'logged.json').exists() else 0
            if logged < ck['global_step']:
                metrics_path = root / 'steps' / f"{ck['global_step']:08d}.metrics.json"
                saved_metrics = read_json(metrics_path) if metrics_path.exists() else ck['metrics']
                log_step(logger, root, ck['global_step'], saved_metrics)
    root_call(report, rank)
    total = c['epochs'] * ((len(selected) + c['prompts_per_step'] - 1) // c['prompts_per_step'])
    total = min(total, c.get('max_steps') or total)
    with tqdm(total=total, initial=ck['global_step'], desc='Online GRPO', disable=rank != 0 or not c.get('progress', True)) as progress:
        while ck['epoch'] < c['epochs'] and ck['global_step'] < total:
            started = monotonic()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            step = ck['global_step'] + 1
            batch = selected[ck['cursor']:ck['cursor'] + c['prompts_per_step']]
            version = ck['policy_version']
            fingerprint = state_hash(policy.initial_state)
            hashes = rank_call(lambda: fingerprint)
            if len(set(hashes)) != 1:
                raise RuntimeError('Ranks have different policy states')
            pending_path = root / 'pending.json'
            pending = root_call(lambda: read_json(pending_path) if pending_path.exists() else None, rank)
            contract = {'step_id': step, 'policy_version': version, 'state_version': fingerprint,
                        'prompt_ids': [p['id'] for p in batch], 'train_id': train_id}
            if pending and pending['contract']['step_id'] < step:
                pending = None  # Previous step committed before cleanup.
            if pending and pending['contract'] != contract:
                raise ValueError('Pending rollout belongs to a different policy/data step')
            rollout_started = monotonic()
            reused_rollout = pending is not None
            if pending is None:
                def collect():
                    jobs = [(p, g) for i, p in enumerate(batch) if i % world == rank for g in range(c['group_size'])]
                    seeds = [int(digest([c.get('seed', 42), step, version, p['id'], g])[:15], 16) for p, g in jobs]
                    generated = rollout_batch(policy, [p['prompt_ids'] for p, _ in jobs], tokenizer, c['sampling'], seeds)
                    return [{**sample, **contract, 'prompt_id': p['id'], 'prompt_ids': p['prompt_ids'],
                             'prompt': p['prompt'], 'reference': p['reference'], 'sample_index': g,
                             'seed': seed} for (p, g), sample, seed in zip(jobs, generated, seeds)]
                gathered = rank_call(collect)
                samples = [s for local in gathered for s in local]
                order = {p['id']: i for i, p in enumerate(batch)}
                samples.sort(key=lambda s: (order[s['prompt_id']], s['sample_index']))
                if len(samples) != len(batch) * c['group_size']:
                    raise ValueError('Incomplete rollout step')
                rollout_seconds = monotonic() - rollout_started
                pending = {'contract': contract, 'samples': samples, 'rollout_seconds': rollout_seconds}
                root_call(lambda: atomic_json(pending_path, pending), rank)
            else:
                samples = pending['samples']
                rollout_seconds = pending['rollout_seconds']
            expected = {(p['id'], g) for p in batch for g in range(c['group_size'])}
            if len(samples) != len(expected) or {(s['prompt_id'], s['sample_index']) for s in samples} != expected:
                raise ValueError('Incomplete/duplicate step trajectories')
            if any(any(s[k] != contract[k] for k in ('step_id', 'policy_version', 'state_version', 'train_id')) for s in samples):
                raise ValueError('Trajectory policy version mismatch')
            rollout_phase_seconds = monotonic() - rollout_started
            requested_scores = sum('judge' not in s for s in samples)
            judge_started = monotonic()
            samples = score_samples(
                samples, judge, c['judge_concurrency'], lambda: atomic_json(pending_path, pending),
                rank=rank, max_rounds=c.get('judge_retry_rounds', 3),
                retry_delay=c.get('judge_retry_delay', 2.),
                on_retry=lambda attempt, remaining: tqdm.write(
                    f"Step {step}: Judge retry round {attempt + 1}, {remaining} missing scores; successes preserved"),
            )
            judge_seconds = monotonic() - judge_started
            zero = group_advantages(samples, c['group_size'], c.get('advantage_eps', 1e-6))
            train_started = monotonic()
            metrics = train_step(policy, model, optimizer, samples, c, rank, world)
            if rank == 0 and metrics['alignment/max_error_exceeded']:
                tqdm.write(f"Step {step}: BF16 alignment tail max={metrics['alignment/max_abs_logprob_error']:.6f} "
                           f"(diagnostic), ratio={metrics['training/ratio_mean']:.6f}, "
                           f"clip_fraction={metrics['training/clip_fraction']:.4%}")
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            train_seconds = monotonic() - train_started
            ck['global_step'] = step
            ck['policy_version'] += 1 - metrics['training/skipped_update']
            ck['cursor'] += len(batch)
            epoch = ck['epoch']
            if ck['cursor'] == len(selected):
                ck['cursor'] = 0
                ck['epoch'] += 1
            times = rank_call(lambda: {'training': train_seconds, 'total': monotonic() - started,
                                      'memory': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0})
            step_seconds = max(t['total'] for t in times)
            generated_tokens = sum(len(s['completion_ids']) for s in samples)
            metrics.update(sample_metrics(samples))
            metrics.update({'reward/zero_variance_group_fraction': zero, 'rollout/generated_tokens': generated_tokens,
                            'rollout/tokens_per_second': generated_tokens / rollout_seconds,
                            'judge/throughput': requested_scores / judge_seconds,
                            'judge/requested_scores': requested_scores, 'judge/cached_scores': len(samples) - requested_scores,
                            'rollout/reused': int(reused_rollout), 'rollout/generation_seconds': rollout_seconds,
                            'performance/rollout_seconds': rollout_phase_seconds, 'performance/judge_seconds': judge_seconds,
                            'performance/training_seconds': max(t['training'] for t in times),
                            'performance/step_seconds': step_seconds, 'performance/tokens_per_second': generated_tokens / step_seconds,
                            'performance/gpu_peak_memory_bytes': max(t['memory'] for t in times),
                            'policy/global_step': step, 'policy/epoch': epoch, 'policy/version': ck['policy_version'],
                            'policy/rollout_version': version})
            root_call(lambda: atomic_jsonl(root / 'steps' / f'{step:08d}.jsonl', samples), rank)
            root_call(lambda: atomic_json(root / 'steps' / f'{step:08d}.metrics.json', metrics), rank)
            # Durable commit precedes logging/cleanup. A crash before this point replays
            # from the PRE-step optimizer/state/RNG; a crash after it never updates twice.
            checkpoint_started = monotonic()
            commit(metrics)
            final_times = rank_call(lambda: (monotonic() - started, monotonic() - checkpoint_started))
            step_seconds = max(t[0] for t in final_times)
            metrics.update({'performance/step_seconds': step_seconds,
                            'performance/checkpoint_seconds': max(t[1] for t in final_times),
                            'performance/tokens_per_second': generated_tokens / step_seconds})
            root_call(lambda: atomic_json(root / 'steps' / f'{step:08d}.metrics.json', metrics), rank)
            root_call(report, rank)
            root_call(lambda: pending_path.unlink(missing_ok=True), rank)
            progress.update(1)
            progress.set_postfix(loss=f"{metrics['training/loss']:.4f}", reward=f"{metrics['reward/mean']:.3f}")
            if c.get('stop_after_steps') and step >= c['stop_after_steps']:
                break
    if rank == 0 and logger is not None:
        logger.finish()
    return ck


def run(c):
    validate(c)
    local = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if not torch.cuda.is_available() or local >= torch.cuda.device_count() or int(os.environ.get('LOCAL_WORLD_SIZE', 1)) > torch.cuda.device_count():
        raise RuntimeError('One distinct CUDA device per local rank is required')
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=c.get('distributed_timeout_seconds', 300)))
    rank = dist.get_rank() if distributed() else 0
    try:
        with ExitStack() as stack:
            root = Path(c['output'])
            root_call(lambda: root.mkdir(parents=True, exist_ok=True), rank)
            root_call(lambda: stack.enter_context(lock(root / '.training.lock')), rank)
            random.seed(c.get('seed', 42))
            torch.manual_seed(c.get('seed', 42))
            loaded = []
            rank_call(lambda: loaded.append(load_policy(c, torch.device('cuda', local))))
            policy, snapshot = loaded[0]
            tokenizers = []
            rank_call(lambda: tokenizers.append(TRIE_TOKENIZER(c['vocab'])))
            tokenizer = tokenizers[0]
            def select():
                rows = list(prompts(c['data'], tokenizer, c.get('max_prompt_tokens', 8192)))
                if c.get('shuffle', True):
                    random.Random(c.get('seed', 42)).shuffle(rows)
                return rows[:c.get('num_prompts')] if c.get('num_prompts') else rows
            selected = root_call(select, rank)
            if not selected:
                raise ValueError('No eligible prompts')
            snapshot.update(root_call(lambda: dict(data_sha256=file_hash(c['data']), vocab_sha256=file_hash(c['vocab'])), rank))
            if rank == 0:
                import swanlab
                logger = swanlab
            else:
                logger = None
            judge_holder = []
            root_call(lambda: judge_holder.append(Judge(c['judge'])), rank)
            judge = judge_holder[0] if rank == 0 else None
            online_loop(c, policy, snapshot, selected, tokenizer, judge, logger)
    finally:
        if distributed():
            dist.destroy_process_group()


if __name__ == '__main__':
    run(config_cli(__doc__, resume_options=True))
