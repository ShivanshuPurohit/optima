"""Shared engine-launch context manager used by the eval modules.

Centralizes the spawn-safe, tamper-resistant launch: mark this process as the
driver (so it never imports miner code), set the seam env, build the sglang
Engine, and clean it up. Both the KL eval and the benchmark eval use this.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


@contextmanager
def env(**overrides: str):
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


@contextmanager
def launched_engine(cfg, *, bundle_path: str, active: bool):
    """Launch a sglang Engine with the Optima seam configured.

    ``cfg`` is an ``EvalConfig`` (see optima.eval.throughput_kl). The miner
    kernel runs only in the spawned scheduler child; THIS process is marked as
    the driver so it never imports miner code (timing stays tamper-resistant).
    """
    from optima import seam

    seam.mark_driver()
    with env(
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
            yield engine
        finally:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001
                pass
