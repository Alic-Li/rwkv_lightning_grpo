import torch
import pytest
from grpo.kernel import StatePassing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(s, *xs):
    r, w, k, v, a, b = [x.float() for x in xs]
    ys = []
    for t in range(r.shape[1]):
        sa = (s * a[:, t, :, None, :]).sum(-1)
        decay = torch.exp(-0.6065306597 * torch.sigmoid(w[:, t]))
        s = (
            s * decay[..., None, :]
            + sa[..., None] * b[:, t, :, None, :]
            + v[:, t, :, :, None] * k[:, t, :, None, :]
        )
        ys.append((s * r[:, t, :, None, :]).sum(-1))
    return torch.stack(ys, 1).bfloat16(), s


@pytest.mark.parametrize("T", [1, 7, 16, 19, 32])
def test_forward_backward_all_inputs(T):
    torch.manual_seed(T)
    s = (torch.randn(2, 2, 64, 64, device="cuda") * 0.1).requires_grad_()
    xs = [
        (torch.randn(2, T, 2, 64, device="cuda") * 0.1).bfloat16().requires_grad_()
        for _ in range(6)
    ]
    # Include nonzero terminal adjoint; this is essential for cross-chunk gradients.
    dy = torch.randn_like(xs[0])
    ds = torch.randn_like(s) * 0.03
    y, end = StatePassing.apply(s, *xs)
    g = torch.autograd.grad((y * dy).sum() + (end * ds).sum(), (s, *xs))
    ry, re = reference(s, *xs)
    rg = torch.autograd.grad((ry * dy).sum() + (re * ds).sum(), (s, *xs))
    torch.testing.assert_close(y, ry, atol=0.001, rtol=0.01)
    torch.testing.assert_close(end, re, atol=2e-5, rtol=2e-4)
    for j, (a, b) in enumerate(zip(g, rg)):
        torch.testing.assert_close(
            a, b, atol=0.002, rtol=0.035, msg=f"gradient {j}, T={T}"
        )


def test_chunk_state_gradient_and_stream():
    torch.manual_seed(8)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        s = torch.randn(1, 1, 64, 64, device="cuda", requires_grad=True)
        xs = [
            (torch.randn(1, 21, 1, 64, device="cuda") * 0.1).bfloat16().requires_grad_()
            for _ in range(6)
        ]
        y, end = StatePassing.apply(s, *xs)
        whole = torch.autograd.grad(
            y.float().square().sum() + end.square().sum(), s, retain_graph=True
        )[0]
        y1, s1 = StatePassing.apply(s, *[x[:, :7].contiguous() for x in xs])
        y2, s2 = StatePassing.apply(s1, *[x[:, 7:].contiguous() for x in xs])
        parts = torch.autograd.grad(
            torch.cat([y1, y2], 1).float().square().sum() + s2.square().sum(), s
        )[0]
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(end, s2, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(whole, parts, atol=2e-5, rtol=1e-4)
