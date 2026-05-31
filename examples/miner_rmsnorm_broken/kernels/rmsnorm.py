"""Adversarial RMSNorm kernel: skips the normalization to run faster.

Computes ``x * weight`` instead of ``rmsnorm(x) * weight`` — it drops the
``rsqrt(mean(x^2)+eps)`` factor. Faster (no reduction), passes the static scan and
manifest, but the unnormalized activations blow up the model. The end-to-end
benchmark gate must reject it. Not a real submission.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _broken_kernel(x_ptr, w_ptr, out_ptr, eps, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * w  # BROKEN: no rsqrt(mean(x^2)+eps) normalization
    tl.store(out_ptr + row * row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    x2 = x.reshape(-1, x.shape[-1])
    o2 = out.reshape(-1, out.shape[-1])
    rows, n = x2.shape
    BLOCK = triton.next_power_of_2(n)
    _broken_kernel[(rows,)](x2, weight, o2, float(eps), n, x2.stride(0), BLOCK=BLOCK)
