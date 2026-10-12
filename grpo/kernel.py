"""Autograd binding for the project's FP32-state / BF16-activation CUDA op."""
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load

_LOADED = False
CHUNK = 16


def load_kernel():
    global _LOADED
    if not _LOADED:
        root = Path(__file__).resolve().parents[1] / 'cuda'
        os.environ.setdefault('MAX_JOBS', '4')
        load(name='rwkv7_statepassing_grpo',
             sources=[str(root / ('rwkv7_statepassing_clampw' + ext)) for ext in ('.cpp', '.cu')],
             extra_cflags=['-O3'],
             extra_cuda_cflags=['-O3', '--use_fast_math', '-D_N_=64', f'-D_CHUNK_LEN_={CHUNK}'],
             is_python_module=False)
        _LOADED = True


class StatePassing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s0, r, w, k, v, a, b):
        load_kernel()
        B, T, H, N = r.shape
        if N != 64 or T < 1:
            raise ValueError('CUDA requires head_size=64 and T>0')
        xs = (r, w, k, v, a, b)
        if not all(x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
                   and x.shape == r.shape and x.device == r.device for x in xs):
            raise ValueError('Expected contiguous, same-shape BF16 CUDA activations')
        if s0.shape != (B, H, N, N) or s0.dtype != torch.float32 or s0.device != r.device:
            raise ValueError('Expected FP32 [B,H,value,key] initial state on activation device')
        s0 = s0.contiguous()
        y = torch.empty_like(v)
        sT = torch.empty_like(s0)
        s = torch.empty(B, H, (T + CHUNK - 1)//CHUNK, N, N, device=r.device, dtype=torch.float32)
        sa = torch.empty_like(r, dtype=torch.float32)
        with torch.cuda.device(r.device):
            torch.ops.rwkv7_statepassing_clampw.forward(s0, *xs, y, sT, s, sa)
        ctx.save_for_backward(*xs, s, sa)
        ctx.set_materialize_grads(False)
        return y, sT

    @staticmethod
    def backward(ctx, dy, dsT):
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        B, _, H, N = r.shape
        if dy is None:
            dy = torch.zeros_like(v)
        if dsT is None:
            dsT = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float32)
        ds0 = torch.empty_like(dsT)
        grads = [torch.empty_like(x) for x in (r, w, k, v, a, b)]
        with torch.cuda.device(r.device):
            torch.ops.rwkv7_statepassing_clampw.backward(
                r, w, k, v, a, b, dy.contiguous(), dsT.contiguous(), s, sa, ds0, *grads)
        return ds0, *grads


def wkv(s0, r, w, k, v, a, b):
    B, T, C = r.shape
    xs = [x.reshape(B, T, C//64, 64).contiguous() for x in (r,w,k,v,a,b)]
    y, state = StatePassing.apply(s0.contiguous(), *xs)
    return y.reshape(B,T,C), state
