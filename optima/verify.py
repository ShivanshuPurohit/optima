"""Op-level correctness — the cheap gate before any end-to-end eval.

Given a slot and a miner ``entry`` callable, generate deterministic inputs over
the slot's standard shapes, run the miner kernel and the trusted reference, and
compare with an allclose-style tolerance.

This is the per-op analogue of a unit test. It is necessary but NOT sufficient:
small per-op errors that pass here can compound into large end-to-end KL, which
is exactly why the pipeline still runs the end-to-end KL gate afterwards. The
seeds and shapes here are also re-randomized per epoch by the caller so a kernel
cannot special-case the fixed verification inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from optima.slots import SlotSpec


@dataclass
class ShapeResult:
    shape: dict
    dtype: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    detail: str = ""


@dataclass
class VerifyResult:
    slot: str
    dtype: str
    passed: bool
    shape_results: list[ShapeResult]

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.shape_results if not r.passed)


def _compare(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float
) -> tuple[bool, float, float, str]:
    # Returns (passed, max_abs, max_rel, detail) — allclose-style + non-finite guard.
    if actual.shape != expected.shape:
        return False, float("inf"), float("inf"), f"shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}"
    a = actual.float()
    e = expected.float()
    if not torch.isfinite(a).all():
        return False, float("inf"), float("inf"), "actual has non-finite values"
    abs_err = (a - e).abs()
    rel_err = abs_err / (e.abs() + 1e-12)
    # allclose: |a-e| <= atol + rtol*|e|
    slack = atol + rtol * e.abs()
    passed = bool((abs_err <= slack).all())
    return passed, float(abs_err.max()), float(rel_err.max()), ""


def verify_entry(
    slot: SlotSpec,
    entry: Callable[..., None],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    seed: int = 0,
    shapes: Optional[list[dict]] = None,
) -> VerifyResult:
    """Verify a miner ``entry`` against the slot's reference.

    ``entry`` is called as ``entry(*inputs_in_order, out)`` — the same contract
    the dispatcher uses — and must write its result into ``out``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tol = slot.tolerance_for(dtype)
    test_shapes = shapes if shapes is not None else list(slot.shapes)

    results: list[ShapeResult] = []
    for i, shape in enumerate(test_shapes):
        inputs = slot.make_inputs(dtype=dtype, device=device, seed=seed + i, **shape)

        expected = slot.invoke_reference(inputs)
        out = torch.empty(slot.out_shape(inputs), dtype=dtype, device=device)
        try:
            slot.invoke_entry(entry, inputs, out)
        except Exception as exc:  # noqa: BLE001 - report kernel failure as a fail
            results.append(
                ShapeResult(shape=shape, dtype=_name(dtype), passed=False,
                            max_abs_err=float("inf"), max_rel_err=float("inf"),
                            detail=f"kernel raised: {type(exc).__name__}: {exc}")
            )
            continue

        passed, max_abs, max_rel, detail = _compare(out, expected, atol=tol.atol, rtol=tol.rtol)
        results.append(
            ShapeResult(shape=shape, dtype=_name(dtype), passed=passed,
                        max_abs_err=max_abs, max_rel_err=max_rel, detail=detail)
        )

    return VerifyResult(
        slot=slot.name,
        dtype=_name(dtype),
        passed=all(r.passed for r in results) and len(results) > 0,
        shape_results=results,
    )


def _name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def format_verify(result: VerifyResult) -> str:
    lines = [f"[{'PASS' if result.passed else 'FAIL'}] {result.slot} dtype={result.dtype}"]
    for r in result.shape_results:
        status = "ok " if r.passed else "FAIL"
        lines.append(
            f"  {status} shape={r.shape} max_abs={r.max_abs_err:.3e} max_rel={r.max_rel_err:.3e}"
            + (f"  {r.detail}" if r.detail else "")
        )
    return "\n".join(lines)
