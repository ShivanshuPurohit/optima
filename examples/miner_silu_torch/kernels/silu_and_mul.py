"""CPU dry-run kernel for the activation.silu_and_mul slot (pure torch).

Same contract as the Triton example: ``silu_and_mul(x, out)`` writes
``silu(x[..., :d]) * x[..., d:]`` into the validator-allocated ``out``. Exists so
the manifest -> scan -> load -> op-correctness path can be tested without a GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    result = F.silu(x[..., :d].float()).to(x.dtype) * x[..., d:]
    out.copy_(result)
