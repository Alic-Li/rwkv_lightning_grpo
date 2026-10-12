import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def state_hash(state):
    return hashlib.sha256(
        state.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    files = [root / "grpo" / f for f in ("model.py", "kernel.py")] + sorted(
        (root / "cuda").glob("*")
    )
    return digest({p.name: file_hash(p) for p in files if p.is_file()})


def load_policy(config, device):
    from .model import Backbone, StatePolicy

    backbone = Backbone(config["model"], device)
    model_hash = file_hash(config["model"])
    state_path = config.get("state")
    if state_path in (None, ""):
        initial_state = torch.zeros(
            backbone.n_layer,
            backbone.n_head,
            backbone.head_size,
            backbone.head_size,
            dtype=torch.float32,
        )
        version = state_hash(initial_state)
        if int(os.environ.get("RANK", "0")) == 0:
            print("Initial WKV State: FP32 zeros (no checkpoint)", flush=True)
    else:
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
        if ckpt.get("layout") != "layer,head,value,key":
            raise ValueError("Unsupported state layout; import through grpo.init_state")
        if not ckpt.get("complete", True):
            raise ValueError(
                "Use a completed state checkpoint for a new policy snapshot"
            )
        initial_state = ckpt["initial_state"]
        version = state_hash(initial_state)
        if ckpt["backbone_sha256"] != model_hash or ckpt["state_version"] != version:
            raise ValueError("State checkpoint fingerprint mismatch")
    policy = StatePolicy(backbone, initial_state)
    snapshot = {
        "backbone_sha256": model_hash,
        "state_version": version,
        "implementation_sha256": implementation_hash(),
        "precision": "bf16-forward-fp32-state",
    }
    snapshot["policy_version"] = digest(snapshot)
    return policy, snapshot


def config_cli(description, *, resume_options=False):
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    if resume_options:
        p.add_argument("--resume", action="store_true", help="Resume the committed state and pending step")
        p.add_argument("--stop-after-steps", type=int, help="Stop once this total global step is committed")
    args = p.parse_args()
    config = read_json(args.config)
    if resume_options:
        if args.resume:
            config["resume"] = True
        if args.stop_after_steps is not None:
            if args.stop_after_steps < 1:
                p.error("--stop-after-steps must be positive")
            config["stop_after_steps"] = args.stop_after_steps
    return config
