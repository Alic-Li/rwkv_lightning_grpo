"""Create explicit State_v0 or import upstream time_state tensors tensors."""

import argparse
import torch
from .io import atomic_torch, file_hash, state_hash


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
        s = torch.stack([raw[f"blocks.{i}.att.time_state"].float() for i in range(L)])
        if args.layout == "key-value":
            s = s.transpose(-1, -2).contiguous()
        if s.shape != (L, H, N, N):
            raise ValueError("Imported state does not match backbone")
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
