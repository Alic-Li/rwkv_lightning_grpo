"""Training-only activation recomputation; the policy equations stay unchanged."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TrainingPolicy(nn.Module):
    def __init__(self, policy, activation_checkpointing=True):
        super().__init__()
        self.policy = policy
        self.activation_checkpointing = activation_checkpointing

    def _chunk(self, ids, targets, states, shifts):
        logits, states, shifts = self.policy.backbone(ids, states, shifts)
        logp = logits.float().log_softmax(-1)
        return logp.gather(-1, targets[..., None]).squeeze(-1), states, shifts

    def forward(self, ids, targets, chunk_size=128):
        if not self.activation_checkpointing or not torch.is_grad_enabled():
            return self.policy(ids, targets, chunk_size)
        states, shifts = self.policy.start(ids.shape[0]), None
        outputs = []
        for start in range(0, ids.shape[1], chunk_size):
            logp, states, shifts = checkpoint(
                self._chunk,
                ids[:, start : start + chunk_size],
                targets[:, start : start + chunk_size],
                states,
                shifts,
                use_reentrant=False,
                preserve_rng_state=False,  # No dropout or sampling in the backbone.
            )
            outputs.append(logp)
        # Boundary WKV and shift tensors stay attached, including prompt chunks.
        return torch.cat(outputs, dim=1)
