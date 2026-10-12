"""Exercise the actual online engine on two Gloo ranks, including resume."""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from grpo import train
from tests.test_online import FakeJudge, FakeLogger, TinyPolicy, config, fake_rollout, loop, rows


def main():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    root = Path(sys.argv[1])
    train.rollout_batch = fake_rollout
    c = config(root / 'distributed', prompts_per_step=3, epochs=2)
    ck = loop({**c, 'stop_after_steps': 1}, TinyPolicy(), logger=FakeLogger() if rank == 0 else None, selected=rows(5))
    assert ck['global_step'] == 1
    resumed = loop({**c, 'resume': True}, TinyPolicy(), logger=FakeLogger() if rank == 0 else None, selected=rows(5))
    assert resumed['global_step'] == resumed['policy_version'] == 4
    assert resumed['epoch'] == 2
    states = [None, None]
    dist.all_gather_object(states, resumed['initial_state'])
    torch.testing.assert_close(states[0], states[1])
    dist.destroy_process_group()
    if rank == 0:
        reference = loop(config(root / 'single', prompts_per_step=3, epochs=2), TinyPolicy(), selected=rows(5))
        torch.testing.assert_close(resumed['initial_state'], reference['initial_state'])
        for k in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(resumed['optimizer']['state'][0][k], reference['optimizer']['state'][0][k])
        import json
        traces = [json.loads(line) for p in sorted((root / 'distributed/steps').glob('*.jsonl')) for line in p.read_text().splitlines()]
        assert len(traces) == 20
        assert len({(s['step_id'], s['prompt_id'], s['sample_index']) for s in traces}) == 20
        print('ONLINE_DDP_PASS: groups, tails, resume, single-rank equivalence')


if __name__ == '__main__':
    main()
