"""The validator-owned dispatcher.

This is the only place a miner kernel is actually invoked during serving. It is
written so that the *validator* owns everything risky around the call:

  * output allocation (shape/dtype/device/stride) — never the miner,
  * eligibility gating via the registry,
  * a fallback to the trusted baseline on ineligibility or error,
  * a single, auditable call into the miner ``entry(*inputs, out)``.

The miner's ``entry`` therefore only ever sees already-allocated tensors and is
expected to fill ``out``. That is the smallest host surface we can give a
Triton/CuteDSL submission while still letting it own the actual computation.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from optima.registry import REGISTRY, KernelRegistry


def _arch_tag(device_index: int = 0) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


def make_silu_and_mul_dispatcher(
    baseline_forward: Callable[[object, torch.Tensor], torch.Tensor],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "activation.silu_and_mul",
) -> Callable[[object, torch.Tensor], torch.Tensor]:
    """Build a replacement for ``SiluAndMul.forward_*``.

    ``baseline_forward`` is the captured original (used for fallback). The
    returned function has the same ``(self, x)`` signature.
    """

    def dispatched(self: object, x: torch.Tensor) -> torch.Tensor:
        last_dim = x.shape[-1]
        impl = registry.lookup(
            slot,
            dtype_name=_dtype_name(x.dtype),
            last_dim=last_dim,
            arch=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        )
        if impl is None:
            return baseline_forward(self, x)

        d = last_dim // 2
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        try:
            impl.entry(x, out)
        except Exception:
            if registry.strict:
                raise
            # Quality/throughput already protect us; a crashing kernel just loses.
            return baseline_forward(self, x)
        return out

    return dispatched


def make_rmsnorm_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "norm.rmsnorm",
) -> Callable[..., object]:
    """Build a replacement for ``RMSNorm.forward_cuda`` / ``forward_native``.

    sglang's RMSNorm has two modes: plain (``residual is None`` -> return normed)
    and fused add+norm (``residual`` given -> return ``(normed, x+residual)``).
    The validator owns the residual add; the miner kernel only ever computes the
    pure rmsnorm: ``entry(x, weight, out, eps)``. Unusual paths fall back to the
    trusted baseline.
    """

    def dispatched(self, x, residual=None, post_residual_addition=None):
        # Rare / fp32 paths -> trusted baseline (keeps the contract simple & safe).
        if post_residual_addition is not None or getattr(self, "fp32_residual", False):
            return baseline_forward(self, x, residual, post_residual_addition)

        impl = registry.lookup(
            slot,
            dtype_name=_dtype_name(x.dtype),
            last_dim=x.shape[-1],
            arch=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        )
        if impl is None:
            return baseline_forward(self, x, residual, post_residual_addition)

        eps = float(self.variance_epsilon)
        weight = self.weight.data
        try:
            if residual is None:
                out = torch.empty_like(x)
                impl.entry(x, weight, out, eps)
                return out
            new_residual = x + residual  # validator owns the add
            out = torch.empty_like(new_residual)
            impl.entry(new_residual, weight, out, eps)
            return out, new_residual
        except Exception:
            if registry.strict:
                raise
            return baseline_forward(self, x, residual, post_residual_addition)

    return dispatched


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, str(dtype).replace("torch.", ""))
