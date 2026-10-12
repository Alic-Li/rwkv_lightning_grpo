"""Convert and export standalone RWKV SFT time_state checkpoints."""

import argparse
import torch
from .io import atomic_torch, file_hash, state_hash


def import_time_state(raw, shape, layout):
    """Convert explicit source axes to native FP32 [layer,head,value,key]."""
    if layout not in ('value-key', 'key-value'):
        raise ValueError('Raw SFT state requires state_layout=value-key or key-value; square tensor shapes cannot determine axes')
    layers, heads, values, keys = shape
    expected = {f'blocks.{i}.att.time_state' for i in range(layers)}
    if not isinstance(raw, dict) or set(raw) != expected:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(f'Invalid SFT state layers: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}')
    tensors = []
    for i in range(layers):
        tensor = raw[f'blocks.{i}.att.time_state']
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tuple(tensor.shape) != (heads, values, keys):
            raise ValueError(f'Imported state does not match backbone at layer {i}')
        tensors.append(tensor.detach().float())
    state = torch.stack(tensors)
    if layout == 'key-value':
        state = state.transpose(-1, -2).contiguous()
    if not torch.isfinite(state).all():
        raise ValueError('Nonfinite imported state')
    return state


def export_time_state(state):
    """Same keys, BF16 dtype and value-key layout as rwkv_lightning_cuda export."""
    if state.ndim != 4 or state.shape[-2:] != (64, 64) or min(state.shape) < 1:
        raise ValueError('Expected state shape [layer,head,64,64]')
    if not torch.isfinite(state).all():
        raise ValueError('Nonfinite state export')
    bf16 = state.detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
    if not torch.isfinite(bf16).all():
        raise ValueError('BF16 state export overflow')
    return {f'blocks.{i}.att.time_state': bf16[i].contiguous() for i in range(len(bf16))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--import-state")
    p.add_argument("--layout", choices=["key-value", "value-key"], default="value-key")
    args = p.parse_args()
    w = torch.load(args.model, map_location="cpu", weights_only=True, mmap=True)
    L = max(int(k.split(".")[1]) for k in w if k.startswith("blocks.")) + 1
    H, N = w["blocks.0.att.r_k"].shape
    s = torch.zeros(L, H, N, N, dtype=torch.float32)
    if args.import_state:
        raw = torch.load(args.import_state, map_location="cpu", weights_only=True)
        s = import_time_state(raw, (L, H, N, N), args.layout)
    if not torch.isfinite(s).all():
        raise ValueError("Nonfinite state")
    from pathlib import Path

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    atomic_torch(
        args.output,
        {
            "format_version": 1,
            "initial_state": s,
            "state_version": state_hash(s),
            "backbone_sha256": file_hash(args.model),
            "layout": "layer,head,value,key",
            "source": args.import_state or "explicit-zero",
        },
    )
    print(args.output, state_hash(s))


if __name__ == "__main__":
    main()
