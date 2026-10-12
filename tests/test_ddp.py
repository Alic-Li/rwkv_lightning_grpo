"""Real distributed communication; NCCL requires two physical GPUs."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from grpo.io import atomic_json, read_json


def launch(module, *args, world=2, timeout=90):
    p = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        f'--nproc_per_node={world}', '-m', module, *map(str, args)],
                       capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


def test_two_process_gloo():
    assert 'DDP_GLOO_PASS' in launch('tests.ddp_worker')


def test_online_two_process_gloo(tmp_path):
    assert 'ONLINE_DDP_PASS' in launch('tests.online_ddp_worker', tmp_path)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires two distinct GPUs for NCCL')
def test_two_gpu_real_training(tmp_path):
    path = Path(os.environ.get('GRPO_DDP_CONFIG', 'configs/online.smoke.json'))
    if not path.exists():
        pytest.skip('Set GRPO_DDP_CONFIG to an online small-model/Judge configuration')
    c = read_json(path)
    c.update(output=str(tmp_path / 'nccl'), max_steps=3, resume=False, swanlab_mode='offline')
    cfg = tmp_path / 'config.json'
    atomic_json(cfg, c)
    launch('grpo.train', '--config', cfg, timeout=600)
    ck = torch.load(Path(c['output']) / 'latest.pt', weights_only=True)
    assert ck['global_step'] == 3


def test_judge_ddp_waits_and_retries_do_not_timeout(tmp_path):
    assert 'JUDGE_DDP_HEARTBEAT_PASS' in launch('tests.judge_ddp_worker', tmp_path / 'judge.json')
    assert read_json(tmp_path / 'judge.json')['success']
