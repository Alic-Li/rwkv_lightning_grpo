"""The actual dataset has alternating text/train segments, not chat messages."""

import json
import math
import torch


def prompts(path, tokenizer, max_tokens):
    with open(path, encoding="utf-8") as stream:
        for row, line in enumerate(stream):
            obj = json.loads(line)
            if set(obj) != {"segments"} or not isinstance(obj["segments"], list):
                raise ValueError(f"Unexpected dataset structure on line {row + 1}")
            prefix = ""
            for turn, s in enumerate(obj["segments"]):
                if (
                    set(s) != {"text", "train"}
                    or type(s["train"]) is not bool
                    or not isinstance(s["text"], str)
                ):
                    raise ValueError(f"Invalid segment {row}:{turn}")
                if s["train"]:
                    ids = tokenizer.encode(prefix)
                    if ids and len(ids) <= max_tokens:
                        # Preserve source bytes: no chat-template reformat or future tool evidence.
                        yield {
                            "id": f"{row:06d}-{turn:03d}",
                            "row": row,
                            "segment": turn,
                            "prompt": prefix,
                            "prompt_ids": ids,
                            "reference": s["text"],
                        }
                prefix += s["text"]


def advantages(rewards, eps=1e-6):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("advantage eps must be positive and finite")
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("A complete finite group with G>=2 is required")
    r = torch.tensor(rewards, dtype=torch.float64)
    mean = r.mean()
    std = r.std(correction=0)
    return mean.item(), std.item(), ((r - mean) / (std + eps)).tolist()


def collate(samples, device):
    length = max(len(s["prompt_ids"]) + len(s["completion_ids"]) - 1 for s in samples)
    B = len(samples)
    ids = torch.zeros(B, length, dtype=torch.long, device=device)
    targets = torch.zeros_like(ids)
    old = torch.zeros(B, length, device=device)
    mask = torch.zeros_like(old)
    adv = torch.zeros(B, 1, device=device)
    for i, s in enumerate(samples):
        p, c = s["prompt_ids"], s["completion_ids"]
        if not p or not c or not len(c) == len(s["old_logprobs"]) == len(s["mask"]):
            raise ValueError("Invalid trajectory lengths")
        seq = torch.tensor(p + c, device=device)
        ids[i, : len(seq) - 1] = seq[:-1]
        targets[i, : len(seq) - 1] = seq[1:]
        sl = slice(len(p) - 1, len(seq) - 1)
        old[i, sl] = torch.tensor(s["old_logprobs"], device=device)
        mask[i, sl] = torch.tensor(s["mask"], device=device)
        adv[i] = s["advantage"]
    return ids, targets, old, mask, adv


def ppo_sum(new, old, adv, mask, clip):
    # Mask BEFORE exp so inactive prompt/padding values cannot produce 0*inf NaNs.
    delta = torch.where(mask.bool(), new - old, torch.zeros_like(new))
    ratio = delta.exp()
    loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    return (
        (loss * mask).sum(),
        (ratio * mask).sum(),
        (((ratio - 1).abs() > clip) * mask).sum(),
    )
