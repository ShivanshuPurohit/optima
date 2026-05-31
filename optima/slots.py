"""Typed op-slot catalog — the submission ABI.

A *slot* is a single, narrowly-typed replaceable operation in the fixed model
graph. The validator owns this catalog; a miner may only target a slot that
exists here, and may only provide the small ``entry`` callable described by the
slot's contract. Everything else around the op (tensor allocation, the call
site, the rest of the model) stays validator-owned.

Each slot carries everything the validator needs to verify a submission without
trusting it: a trusted ``reference``, a deterministic input generator, the
standard shapes, per-dtype tolerances, and — because different ops have different
call shapes — explicit ``invoke_reference`` / ``invoke_entry`` so verification
doesn't hard-code one signature. (silu is ``entry(x, out)``; rmsnorm is
``entry(x, weight, out, eps)``.)

Adding a slot is a validator action (a code change here), never a miner action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass(frozen=True)
class SlotSpec:
    name: str  # dotted slot id, e.g. "activation.silu_and_mul"
    entry: str  # required callable name the miner module must expose
    summary: str  # human-readable contract

    make_inputs: Callable[..., dict]  # (**shape, dtype, device, seed) -> {name: tensor|scalar}
    out_shape: Callable[[dict], tuple]  # (inputs) -> output shape
    invoke_reference: Callable[[dict], torch.Tensor]  # (inputs) -> expected tensor
    invoke_entry: Callable[..., None]  # (entry, inputs, out) -> None (writes out)
    shapes: tuple[dict, ...]
    tolerances: dict[torch.dtype, Tolerance] = field(default_factory=dict)

    def tolerance_for(self, dtype: torch.dtype) -> Tolerance:
        if dtype in self.tolerances:
            return self.tolerances[dtype]
        if dtype in (torch.float16, torch.bfloat16):
            return Tolerance(atol=2e-2, rtol=2e-2)
        return Tolerance(atol=1e-4, rtol=1e-4)


_BF16_TOL = {
    torch.bfloat16: Tolerance(2e-2, 2e-2),
    torch.float16: Tolerance(1e-2, 1e-2),
    torch.float32: Tolerance(1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Slot: activation.silu_and_mul   (Qwen/Llama-class MLP)
#   x:(...,2d) -> out:(...,d) = silu(x[...,:d]) * x[...,d:]
#   contract: entry(x, out)
# ---------------------------------------------------------------------------


def _silu_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d].float()).to(x.dtype) * x[..., d:]


def _silu_inputs(*, num_tokens: int, d: int, dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, 2 * d, generator=g, device=device, dtype=torch.float32).to(dtype)
    return {"x": x}


SILU_AND_MUL = SlotSpec(
    name="activation.silu_and_mul",
    entry="silu_and_mul",
    summary="out = silu(x[...,:d]) * x[...,d:];  x:(...,2d) -> out:(...,d);  entry(x, out)",
    make_inputs=_silu_inputs,
    out_shape=lambda i: (*i["x"].shape[:-1], i["x"].shape[-1] // 2),
    invoke_reference=lambda i: _silu_reference(i["x"]),
    invoke_entry=lambda entry, i, out: entry(i["x"], out),
    shapes=(
        {"num_tokens": 1, "d": 1024},
        {"num_tokens": 8, "d": 1024},
        {"num_tokens": 128, "d": 4096},
        {"num_tokens": 4096, "d": 4096},
        {"num_tokens": 333, "d": 2880},
    ),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot: norm.rmsnorm   (universal — every transformer, incl. GPT-OSS)
#   out = x / sqrt(mean(x^2, -1) + eps) * weight
#   x:(...,H), weight:(H,) -> out:(...,H)
#   contract: entry(x, weight, out, eps)
# The validator owns the residual add (fused add+norm) and only ever asks the
# miner to compute the pure normalization.
# ---------------------------------------------------------------------------


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps)
    return (normed * weight.float()).to(x.dtype)


def _rmsnorm_inputs(*, num_tokens: int, hidden: int, dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn(hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    return {"x": x, "weight": w, "eps": 1e-6}


RMSNORM = SlotSpec(
    name="norm.rmsnorm",
    entry="rmsnorm",
    summary="out = x*rsqrt(mean(x^2,-1)+eps)*weight;  x:(...,H),weight:(H,) -> out:(...,H);  entry(x, weight, out, eps)",
    make_inputs=_rmsnorm_inputs,
    out_shape=lambda i: tuple(i["x"].shape),
    invoke_reference=lambda i: _rmsnorm_reference(i["x"], i["weight"], i["eps"]),
    invoke_entry=lambda entry, i, out: entry(i["x"], i["weight"], out, i["eps"]),
    shapes=(
        {"num_tokens": 1, "hidden": 2880},
        {"num_tokens": 8, "hidden": 2880},
        {"num_tokens": 128, "hidden": 2880},
        {"num_tokens": 4096, "hidden": 4096},
        {"num_tokens": 333, "hidden": 1536},
    ),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

SLOTS: dict[str, SlotSpec] = {
    SILU_AND_MUL.name: SILU_AND_MUL,
    RMSNORM.name: RMSNORM,
}


def get_slot(name: str) -> SlotSpec:
    try:
        return SLOTS[name]
    except KeyError:
        known = ", ".join(sorted(SLOTS)) or "(none)"
        raise KeyError(f"unknown slot {name!r}; known slots: {known}") from None


def list_slots() -> list[str]:
    return sorted(SLOTS)
