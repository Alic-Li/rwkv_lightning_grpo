"""Real multi-process DDP tests; GPU test automatically runs when two GPUs exist."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from grpo.io import atomic_json, read_json


def test_two_process_gloo():
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "tests.ddp_worker",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "DDP_GLOO_PASS" in p.stdout


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires two distinct GPUs for NCCL"
)
def test_two_gpu_real_training(tmp_path):
    # Uses the complete offline smoke artifact, without sampling or Judge calls.
    config_path = Path(os.environ.get("GRPO_DDP_CONFIG", "configs/stage3.smoke.json"))
    c = read_json(config_path)
    if not Path(c["dataset"]).exists():
        pytest.skip("First export the smoke dataset or set GRPO_DDP_CONFIG")
    outputs = []
    for world in (1, 2):
        dataset = tmp_path / f"dataset{world}"
        dataset.mkdir()
        shutil.copy(Path(c["dataset"]), dataset / "manifest.json")
        for name in ("rollouts.jsonl", "judgments.jsonl"):
            shutil.copy(Path(c["dataset"]).parent / name, dataset / name)
        conf = {
            **c,
            "dataset": str(dataset / "manifest.json"),
            "output": str(tmp_path / f"train{world}"),
            "microbatch": 1,
            "gradient_accumulation": 4 // world,
            "resume": False,
            "swanlab_mode": "offline",
        }
        cfg = tmp_path / f"config{world}.json"
        atomic_json(cfg, conf)
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world}",
                "-m",
                "grpo.stage3",
                "--config",
                str(cfg),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert p.returncode == 0, p.stdout + p.stderr
        outputs.append(
            torch.load(
                Path(conf["output"]) / "latest.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    assert outputs[0]["cursor"] == outputs[1]["cursor"]
    # BF16 backward accumulation order can differ; compare the actual state update.
    torch.testing.assert_close(
        outputs[0]["initial_state"], outputs[1]["initial_state"], atol=3e-5, rtol=0.03
    )

    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            outputs[0]["optimizer"]["state"][0][key],
            outputs[1]["optimizer"]["state"][0][key],
            atol=1e-5,
            rtol=0.05,
        )
