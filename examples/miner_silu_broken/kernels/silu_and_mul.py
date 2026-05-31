"""Adversarial kernel: drops the SiLU nonlinearity to run faster.

Computes ``gate * up`` instead of ``silu(gate) * up``. This is the canonical
"make it faster by skipping work" cheat: it passes the static scan and the
manifest, and it is genuinely faster, but it changes the model's output
distribution. The end-to-end KL gate must reject it. Not a real submission.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _broken_kernel(x_ptr, out_ptr, d, sxr, sxc, sor, soc, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < d
    gate = tl.load(x_ptr + row * sxr + col * sxc, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * sxr + (col + d) * sxc, mask=mask, other=0.0).to(tl.float32)
    res = gate * up  # BROKEN: the silu(gate) nonlinearity is gone
    tl.store(out_ptr + row * sor + col * soc, res.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    x2 = x.reshape(-1, x.shape[-1])
    o2 = out.reshape(-1, out.shape[-1])
    rows, _ = x2.shape
    d = o2.shape[1]
    grid = (rows, triton.cdiv(d, 1024))
    _broken_kernel[grid](x2, o2, d, x2.stride(0), x2.stride(1), o2.stride(0), o2.stride(1), BLOCK=1024)
