"""Judge waits exceed the process-group timeout; one-second heartbeats keep it alive."""
import sys
import time
from collections import Counter
from datetime import timedelta

import torch.distributed as dist

from grpo.io import atomic_json
from grpo.online import score_samples


def main():
    dist.init_process_group('gloo', timeout=timedelta(seconds=3))
    rank = dist.get_rank()
    calls = Counter()
    class Judge:
        def score(self, p, c, r):
            calls[c] += 1
            if c == 'slow':
                time.sleep(4)
                if calls[c] == 1:
                    raise RuntimeError('transient server failure')
            return {'reward': .75}
    samples = [dict(prompt='p', reference='r', completion=c) for c in ['fast', 'slow']]
    scored = score_samples(samples, Judge() if rank == 0 else None, 2, lambda: None,
                           rank=rank, max_rounds=2, retry_delay=0)
    assert all(s['reward'] == .75 for s in scored)
    assert len(scored[1]['judge_failures']) == 1
    if rank == 0:
        assert calls == {'fast': 1, 'slow': 2}
        atomic_json(sys.argv[1], dict(calls=calls, success=True))
        print('JUDGE_DDP_HEARTBEAT_PASS')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
