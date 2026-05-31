"""Benchmark-based capability eval — the score is throughput, gated by task
performance on real benchmarks.

This implements the design: the quality gate is not (only) KL — it's "did the
model's accuracy on real benchmark problems survive the kernel?" We run the same
fixed benchmark sample through two launches (kernel off = baseline, kernel on =
candidate), measure throughput and per-benchmark accuracy for each, and gate:

* **quality**: the candidate must not regress on ANY benchmark beyond a small
  tolerance (Affine's "strictly not worse across all envs", applied to accuracy).
* **score**: the single thing maximized is THROUGHPUT speedup. The benchmarks are
  pass/fail GATES, not score components — so there's nothing to aggregate with a
  geometric mean. Our objective is scalar (speed); correctness is a constraint.

A faithful kernel preserves accuracy and (hopefully) speeds things up. A kernel
that secretly degrades the model drops benchmark accuracy and scores zero, even
if it looked fast.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch

from optima.eval._launch import launched_engine
from optima.eval.benchmarks import Problem, get_benchmark
from optima.eval.throughput_kl import EvalConfig


@dataclass
class BenchmarkScore:
    name: str
    n: int
    baseline_correct: int
    candidate_correct: int

    @property
    def baseline_acc(self) -> float:
        return self.baseline_correct / self.n if self.n else 0.0

    @property
    def candidate_acc(self) -> float:
        return self.candidate_correct / self.n if self.n else 0.0

    @property
    def delta(self) -> float:
        return self.candidate_acc - self.baseline_acc


@dataclass
class CapabilityReport:
    benchmarks: list[BenchmarkScore]
    baseline_tok_s: float
    candidate_tok_s: float
    speedup: float
    passed_quality: bool
    passed_speedup: bool
    score: float
    regressions: list[str] = field(default_factory=list)


def _generate_and_time(engine, prompts: list[str], *, max_new_tokens: int, timed_iters: int):
    sp = {"temperature": 0.0, "max_new_tokens": max_new_tokens}
    # warmup (JIT/compile off the clock)
    engine.generate(prompt=prompts, sampling_params=sp)

    samples: list[float] = []
    outputs = None
    for i in range(max(1, timed_iters)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = engine.generate(prompt=prompts, sampling_params=sp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if isinstance(outs, dict):
            outs = [outs]
        tokens = sum(int(o.get("meta_info", {}).get("completion_tokens", 0)) for o in outs)
        if elapsed > 0:
            samples.append(tokens / elapsed)
        outputs = outs  # keep last for answer-checking
    tok_s = statistics.median(samples) if samples else 0.0
    texts = [o.get("text", "") for o in (outputs or [])]
    return tok_s, texts


def _run_launch(cfg: EvalConfig, flat: list[tuple[str, Problem]], *, bundle_path: str, active: bool,
                max_new_tokens: int):
    prompts = [p.prompt for _, p in flat]
    with launched_engine(cfg, bundle_path=bundle_path, active=active) as engine:
        return _generate_and_time(engine, prompts, max_new_tokens=max_new_tokens,
                                  timed_iters=cfg.timed_iters)


def _accuracy_by_benchmark(flat: list[tuple[str, Problem]], texts: list[str]) -> dict[str, int]:
    correct: dict[str, int] = {}
    for (bench_name, problem), text in zip(flat, texts):
        bench = get_benchmark(bench_name)
        if bench.check(problem, text):
            correct[bench_name] = correct.get(bench_name, 0) + 1
        else:
            correct.setdefault(bench_name, 0)
    return correct


def evaluate_capability(
    cfg: EvalConfig,
    bundle_path: str,
    benchmark_names: list[str],
    *,
    samples_per_benchmark: int = 32,
    acc_tolerance: float = 0.02,
) -> CapabilityReport:
    # Build one flat, ordered list of (benchmark, problem); max_new_tokens is the
    # max any benchmark needs (they're generated together).
    flat: list[tuple[str, Problem]] = []
    counts: dict[str, int] = {}
    max_new = 0
    for name in benchmark_names:
        bench = get_benchmark(name)
        probs = bench.load(samples_per_benchmark, cfg.prompt_seed)
        counts[name] = len(probs)
        max_new = max(max_new, bench.max_new_tokens)
        flat.extend((name, p) for p in probs)

    base_tok_s, base_texts = _run_launch(cfg, flat, bundle_path="", active=False, max_new_tokens=max_new)
    cand_tok_s, cand_texts = _run_launch(cfg, flat, bundle_path=bundle_path, active=True, max_new_tokens=max_new)

    base_correct = _accuracy_by_benchmark(flat, base_texts)
    cand_correct = _accuracy_by_benchmark(flat, cand_texts)

    scores: list[BenchmarkScore] = []
    regressions: list[str] = []
    for name in benchmark_names:
        bs = BenchmarkScore(
            name=name, n=counts[name],
            baseline_correct=base_correct.get(name, 0),
            candidate_correct=cand_correct.get(name, 0),
        )
        scores.append(bs)
        if bs.delta < -acc_tolerance:  # regressed beyond tolerance
            regressions.append(f"{name}: {bs.baseline_acc:.1%} -> {bs.candidate_acc:.1%}")

    speedup = (cand_tok_s / base_tok_s) if base_tok_s > 0 else 0.0
    passed_quality = len(regressions) == 0
    passed_speedup = speedup >= (1.0 + cfg.speedup_margin)
    score = speedup if (passed_quality and passed_speedup) else (0.0 if not passed_quality else speedup)

    return CapabilityReport(
        benchmarks=scores,
        baseline_tok_s=base_tok_s,
        candidate_tok_s=cand_tok_s,
        speedup=speedup,
        passed_quality=passed_quality,
        passed_speedup=passed_speedup,
        score=score,
        regressions=regressions,
    )
