"""End-to-end evaluation: throughput + output-distribution (KL) fidelity.

Two launches of the same model (identical weights/seed/sampler), differing only
by whether the miner kernel is enabled, isolate the kernel's effect: the
throughput delta is the kernel's, and the per-position KL between the two runs is
how much it perturbed the output. A faithful kernel yields KL ~ 0 and (hopefully)
speedup > 1.

Robustness measures (vs the first MVP):

* tamper-resistant timing — the driver process calls ``seam.mark_driver()`` so it
  never imports the miner module; the kernel runs only in the spawned scheduler,
  which the driver times over IPC. A malicious kernel cannot reach the clock.
* median-of-K — each launch times the workload K times and reports the median
  plus spread, so a single noisy sample can't swing the score.
* larger, seeded prompt set — sampled per epoch from a corpus so a kernel can't
  special-case a fixed handful of prompts, and more positions stabilize the KL.

GPU-only; imports sglang lazily.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from optima.eval.kl import KLReport, kl_over_positions
from optima.eval.prompts import sample_prompts


@dataclass
class EvalConfig:
    model_path: str
    dtype: str = "bfloat16"
    max_new_tokens: int = 64
    num_prompts: int = 32
    timed_iters: int = 3  # median-of-K timed passes per launch
    top_logprobs_num: int = 20
    temperature: float = 0.0  # greedy -> deterministic alignment
    warmup_iters: int = 1
    deterministic: bool = False
    kl_threshold: float = 5e-3
    seed: int = 0  # model seed
    prompt_seed: int = 0  # per-epoch prompt sampling seed
    # speedup must clear this margin over 1.0 to count as a real improvement,
    # absorbing measurement noise (see settle/champion logic too).
    speedup_margin: float = 0.02
    attention_backend: str = "triton"
    disable_cuda_graph: bool = True
    mem_fraction_static: float = 0.6
    log_level: str = "warning"
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeResult:
    tok_per_s: float  # median across timed_iters
    tok_per_s_samples: list[float]
    tokens: int
    per_prompt: list[tuple[list[int], list]]  # (output_ids, per-position top-k)

    @property
    def spread(self) -> tuple[float, float, float]:
        s = self.tok_per_s_samples
        if len(s) < 2:
            return (min(s, default=0.0), max(s, default=0.0), 0.0)
        return (min(s), max(s), statistics.pstdev(s))


@dataclass
class EvalReport:
    baseline: ModeResult
    candidate: ModeResult
    speedup: float
    kl: KLReport
    passed_quality: bool
    passed_speedup: bool
    score: float


@contextmanager
def _env(**overrides: str):
    import os

    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _sampling_params(cfg: EvalConfig) -> dict:
    return {"temperature": cfg.temperature, "max_new_tokens": cfg.max_new_tokens}


def _extract_per_prompt(outputs: list[dict]) -> list[tuple[list[int], list]]:
    per_prompt: list[tuple[list[int], list]] = []
    for o in outputs:
        meta = o.get("meta_info", {})
        output_ids = o.get("output_ids") or meta.get("output_ids") or []
        topk = meta.get("output_top_logprobs") or []
        per_prompt.append(([int(t) for t in output_ids], topk))
    return per_prompt


def _timed_generate(engine, prompts: list[str], cfg: EvalConfig, *, with_logprobs: bool):
    sp = _sampling_params(cfg)
    kwargs: dict[str, Any] = {}
    if with_logprobs:
        kwargs = dict(return_logprob=True, logprob_start_len=-1, top_logprobs_num=cfg.top_logprobs_num)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.generate(prompt=list(prompts), sampling_params=sp, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if isinstance(outputs, dict):
        outputs = [outputs]
    tokens = sum(int(o.get("meta_info", {}).get("completion_tokens", 0)) for o in outputs)
    return outputs, tokens, elapsed


def _measure(engine, prompts: list[str], cfg: EvalConfig) -> ModeResult:
    # Warmup (JIT/compile/graph) off the clock.
    for _ in range(max(0, cfg.warmup_iters)):
        engine.generate(prompt=list(prompts), sampling_params=_sampling_params(cfg))

    samples: list[float] = []
    last_outputs = None
    last_tokens = 0
    for i in range(max(1, cfg.timed_iters)):
        # Capture logprobs only on the last iter (cheaper, and the dist is stable).
        with_lp = i == cfg.timed_iters - 1
        outputs, tokens, elapsed = _timed_generate(engine, prompts, cfg, with_logprobs=with_lp)
        if elapsed > 0:
            samples.append(tokens / elapsed)
        if with_lp:
            last_outputs, last_tokens = outputs, tokens

    return ModeResult(
        tok_per_s=statistics.median(samples) if samples else 0.0,
        tok_per_s_samples=samples,
        tokens=last_tokens,
        per_prompt=_extract_per_prompt(last_outputs or []),
    )


def _run_launch(cfg: EvalConfig, prompts: list[str], *, bundle_path: str, active: bool) -> ModeResult:
    # Mark THIS process as the timer/driver before importing sglang, so the seam
    # here is pass-through only and the miner module is never imported in the
    # process that measures wall-clock.
    from optima import seam

    seam.mark_driver()

    with _env(
        OPTIMA_BUNDLE_PATH=bundle_path or "",
        OPTIMA_ACTIVE="1" if active else "0",
        SGLANG_PLUGINS="optima",
    ):
        import sglang as sgl

        kwargs: dict[str, Any] = dict(
            model_path=cfg.model_path,
            dtype=cfg.dtype,
            attention_backend=cfg.attention_backend,
            disable_cuda_graph=cfg.disable_cuda_graph,
            mem_fraction_static=cfg.mem_fraction_static,
            random_seed=cfg.seed,
            log_level=cfg.log_level,
        )
        if cfg.deterministic:
            kwargs["enable_deterministic_inference"] = True
        kwargs.update(cfg.extra_engine_kwargs)
        engine = sgl.Engine(**kwargs)
        try:
            return _measure(engine, prompts, cfg)
        finally:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001
                pass


def _aligned_kl(baseline: ModeResult, candidate: ModeResult) -> KLReport:
    ref_positions: list = []
    cand_positions: list = []
    for (b_ids, b_topk), (c_ids, c_topk) in zip(baseline.per_prompt, candidate.per_prompt):
        n = min(len(b_topk), len(c_topk))
        for i in range(n):
            # Position i is comparable iff the generated prefix 0..i-1 matched
            # (guaranteed here since we break right after the first mismatch).
            # Position 0 always matches (identical prompt), so a kernel that
            # derails the first token still gets scored — compare i, THEN stop.
            ref_positions.append(b_topk[i])
            cand_positions.append(c_topk[i])
            if i < len(b_ids) and i < len(c_ids) and b_ids[i] != c_ids[i]:
                break
    return kl_over_positions(ref_positions, cand_positions)


def evaluate(cfg: EvalConfig, bundle_path: str, prompts: Optional[list[str]] = None) -> EvalReport:
    prompts = list(prompts) if prompts else sample_prompts(cfg.num_prompts, cfg.prompt_seed)

    baseline = _run_launch(cfg, prompts, bundle_path="", active=False)
    candidate = _run_launch(cfg, prompts, bundle_path=bundle_path, active=True)

    kl = _aligned_kl(baseline, candidate)
    speedup = (candidate.tok_per_s / baseline.tok_per_s) if baseline.tok_per_s > 0 else 0.0
    passed_quality = kl.mean_kl <= cfg.kl_threshold and kl.num_positions > 0
    passed_speedup = speedup >= (1.0 + cfg.speedup_margin)
    # Score: the speedup, but only counted as positive when BOTH quality holds and
    # the speedup clears the noise margin. A faithful-but-not-faster kernel scores
    # ~1.0 (no improvement); a cheat scores 0.
    score = speedup if (passed_quality and passed_speedup) else (0.0 if not passed_quality else speedup)

    return EvalReport(baseline, candidate, speedup, kl, passed_quality, passed_speedup, score)
