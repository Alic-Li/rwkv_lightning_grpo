"""torchrun worker: exact uneven-token DDP/accumulation normalization oracle.

CPU/Gloo allows real two-process reducer testing on a one-GPU host.
The CUDA recurrence itself is checked separately against autograd.
"""

from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from grpo.data import ppo_sum


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

        def forward(self, x):
            return x @ self.state

    model = Policy()
    ddp = DDP(model)
    # Last accumulation slot is empty on rank 1; other slots have unequal lengths.
    batches = [
        (torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]]), torch.tensor([1.0, -1.0])),
        (torch.tensor([[-1.0, 0.0, 2.0]]), torch.tensor([0.4])),
        (
            torch.tensor([[3.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 2.0, 1.0]]),
            torch.tensor([-0.5, 0.8, 1.2]),
        ),
    ]
    count = sum(len(x) for x, a in batches)
    for slot in range(2):
        idx = slot * world + rank
        if idx < len(batches):
            x, a = batches[idx]
            mask = torch.ones(len(x))
        else:
            x, a = batches[0]
            mask = torch.zeros(len(x))
        with ddp.no_sync() if slot == 0 else nullcontext():
            new = ddp(x)
            old = (x @ torch.tensor([0.1, -0.2, 0.3])).detach()
            loss, _, _ = ppo_sum(new, old, a, mask, 0.2)
            (loss * world / count).backward()
    grad = model.state.grad.clone()
    reference = Policy()
    x = torch.cat([x for x, a in batches])
    a = torch.cat([a for x, a in batches])
    new = reference(x)
    loss, _, _ = ppo_sum(new, new.detach(), a, torch.ones(count), 0.2)
    (loss / count).backward()
    torch.testing.assert_close(grad, reference.state.grad, atol=1e-7, rtol=1e-6)
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    torch.testing.assert_close(model.state, reference.state, atol=1e-7, rtol=1e-6)
    if rank == 0:
        print(
            "DDP_GLOO_PASS: two ranks, uneven tokens, accumulation, dummy rank; grad=",
            grad.tolist(),
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
