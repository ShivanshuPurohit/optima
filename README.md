# Optima — inference-throughput competition harness on SGLang

A validator harness for a Bittensor-style subnet where miners submit **kernels**
(Triton / CuteDSL) that get swapped into a **fixed** model at typed op-slots, and
are scored on **throughput** gated by **output fidelity** — measured two ways:
per-token **KL** against a reference run, and **task accuracy** on real benchmarks.

> **New here? Read [docs/HOW_OPTIMA_WORKS.md](docs/HOW_OPTIMA_WORKS.md)** — the
> full end-to-end explainer: what the validator does, what miners submit, the
> exact pipeline, how a kernel gets into the model, and the complete threat model
> (including the "fake the output via an API call" attack and why the op-slot
> design defeats it). *(Some of its prose predates the current 7-slot catalog and
> the first throughput win — this README is the current state of record.)*
>
> **Going to production?** [docs/SUBNET_BLUEPRINT.md](docs/SUBNET_BLUEPRINT.md)
> distills how a real Bittensor subnet (Affine) is built — chain plumbing, the
> service decomposition, DB-backed state, copy detection, and the isolation
> security pattern — and maps each onto Optima's production architecture.

## What is and isn't done

**Done & validated on real GPUs (H100, up to gpt-oss-120b; sglang 0.5.12.post1 / CUDA 13):**
the whole *mechanism* — typed op-slots, fused-*block* slots, **and a cross-GPU *collective*
slot** (a slot can be one op, a region behind one typed tensor boundary, or a collective
handed the process group), the seam that swaps an untrusted kernel into a spawned model
process, op-correctness, two-launch throughput measurement, the KL gate, a real-task
capability gate (GSM8K + MMLU), commit-reveal + king-of-the-hill scoring, and
tamper-resistant timing. **Seven slots:** `activation.silu_and_mul`, `norm.rmsnorm` (ops);
`attention.sdpa` / `attention.decode`, `moe.fused_experts` / `moe.fused_experts_mxfp4`
(blocks); `collective.all_reduce` (collective, verified distributed). The
**attention-decode swap is proven end-to-end on a live Qwen** (the validator extracts the
running model's paged KV and routes decode to the miner kernel; a broken kernel is caught
~20×), and scoring runs with **CUDA graphs ON + the hardware's best attention backend**,
never a graphs-off/triton weak baseline.

**First throughput win — through SlotSpec, no fork.** On 4× RTX PRO 6000 Blackwell (sm120)
the `moe.fused_experts_mxfp4` bundle routes gpt-oss-120b's experts through the
`FusedMoE.forward` seam at **~912–922 tok/s (TP=4, batch 32)** — matching the hand-forked
`flashinfer_mxfp4` backend (~926) while the validator runs **stock pinned sglang**, and
**+19% over the stock-realizable path** (stock sglang is forced onto the triton MoE fallback
on sm120). Be honest about what it is: a *route-to-a-faster-existing-path-via-the-seam* win
on a frontier where stock has only a slow fallback — **not** a novel hand-written kernel
beating a mature tuned baseline (that's still open). The silu/rmsnorm/attention demos remain
toy/slow; they prove the contract — the MXFP4 MoE is the first speed win.

**Still open:** a novel kernel beating stock on a mature path, isolation for untrusted
miners, chain integration, a real DB, the next arena (**B200/sm100**, where sglang's FP4 MoE
genuinely works and is tuned), and **eval calibration** (see "Calibration findings").

## Status: validated end-to-end

Two-launch runs (baseline = stock kernels, candidate = miner kernel swapped into
the live model). The **broken** kernels are adversarial — faster-looking but they
degrade the model; the gate must reject them.

**Qwen2.5-1.5B, GSM8K benchmark gate:**

| Bundle | GSM8K base→cand | throughput | gate | score |
|---|---|---|---|---|
| `miner_silu_triton` (faithful) | 62.5% → 62.5% | 0.94× | **PASS** | 1.0 |
| `miner_silu_broken` (drops SiLU) | 62.5% → **0.0%** | 1.26× faster | **FAIL** | **0** |

The cheat is genuinely 26% faster yet scores **zero** because it can't do the
work anymore. *Fast-but-dumb = worthless.*

**gpt-oss-120b (MXFP4, single H100), GSM8K + KL:**

| Bundle | GSM8K base→cand | KL | gate |
|---|---|---|---|
| `miner_rmsnorm_broken` (skips norm) | 75.0% → **0.0%** | huge | **FAIL** (correct) |
| `miner_rmsnorm_triton` (faithful) | 75.0% → 58.3%* | 9.2e-3* | FAIL* |

\* We measured the control — stock-vs-stock KL (the nondeterminism floor) is
**3.9e-4** (1/2041 token flips). The faithful kernel's **9.2e-3 / 24-flips is ~24×
the floor**, so it's *real* drift, not sampling noise: this toy kernel isn't
bit-faithful to sglang's RMSNorm, and the **end-to-end gate correctly caught what
op-correctness (bf16 tolerance) passed**. Wins here: the RMSNorm seam **fires on a
120B MoE model** (gpt-oss fuses its activation into the MoE kernel, so `SiluAndMul`
is inert but `RMSNorm` fires), the cheat is caught hard (75%→0%), and the gate
caught a *subtle* real drift a per-op check missed.

**gpt-oss-120b (MXFP4) on 4× RTX PRO 6000 Blackwell (sm120) — first throughput win, through SlotSpec:**

The `moe.fused_experts_mxfp4` bundle implements GPT-OSS's MXFP4 fused experts (CUTLASS
MXFP8×MXFP4, autotuned) as a `(prepare, forward)` **block** slot, routed in via the
`FusedMoE.forward` seam — no sglang fork. Apples-to-apples, TP=4, batch 32, eager:

| MoE path | tok/s | forks sglang? |
|---|---|---|
| stock sm120 best (triton + CUDA graphs) | 767 | — |
| **seam → MXFP4 kernel (autotuned)** | **912–922** | **no** |
| hand-forked `flashinfer_mxfp4` backend | 926 | yes |

The seam **ties the forked backend (~99%)** while the validator runs stock pinned sglang,
and beats the stock-realizable path by **+19%**. On sm120 stock sglang is *forced* onto the
triton MoE fallback (the `flashinfer_*` MoE backends crash there), so triton isn't a weak
baseline we chose — it's what sglang can run; the seam recovers the optimized backend's speed
without forking. Fidelity is gated by a new **`cosine`** correctness mode (0.985 vs the fp32
reference; element-wise tolerance is meaningless at fp4 per-element error).

Run **eager** with GPU headroom for the first-forward `prepare` (which pads/quantizes the dense
experts): the eval's `mem_fraction_static≈0.6` works as-is; at `0.85` set full-eager
(`disable_piecewise_cuda_graph`) **and** `OPTIMA_MOE_FREE_DENSE=1` (reclaims the dense bf16 experts
after prepare — the kernel owns its MXFP4 copies). The production-clean fix (any mem_fraction) is
load-time weight conversion — tracked. The meaningful next arena is **B200/sm100**, where sglang's
FP4 MoE genuinely works and is heavily tuned.

### Calibration findings (from running on real hardware)

1. **The KL threshold must be calibrated to the model's nondeterminism noise
   floor**, not hand-picked. We measured it on gpt-oss-120b: stock-vs-stock KL with
   `--no-deterministic` is **3.9e-4** (1/2041 flips) — the floor. Set ε = k×floor
   (e.g. 5×), and run with `enable_deterministic_inference` so the floor → ~0 and
   kernel drift is cleanly attributable. (The faithful rmsnorm above sat at 24×
   the floor — genuinely above any sane threshold, correctly flagged.)
2. **Benchmark accuracy needs large n.** At n=12, GSM8K has a ~12% std; a 2-problem
   flip reads as "−16.7%." Use **KL as the dense, low-variance primary gate** and
   **benchmark accuracy as a capability floor at ~100–200 samples**.
3. **For a quantized model there's no fp32 ground truth** (gpt-oss is MXFP4), so the
   KL reference is the stock-kernel run; the threshold must tolerate benign
   rounding in either direction.
4. **Big MoE models need per-launch process isolation + deterministic scoring.**
   The two launches must each run in their **own process** (`call_in_subprocess`):
   on gpt-oss-120b in deterministic mode, running baseline then candidate in one
   driver process corrupted the candidate (NaN outputs → a *no-op* kernel "regressed"
   to 0%). With isolation, deterministic mode works and the stock-vs-stock KL floor
   is **~0** (a clean gate — validated: a no-op scores KL `0.0`, PASS). In
   **non-deterministic** mode the floor on the realistic long-generation workload is
   **1.17e-2** — *above* a 5e-3 gate — so a faithful kernel false-fails. Takeaway:
   **score big MoE in deterministic mode**; where that's unavailable, run
   `--kl-advisory` and let the **accuracy gate** carry quality. (KL is also now
   hardened: a genuinely degenerate candidate — all-non-finite logprobs — reads as
   maximal divergence, not 0.)
5. **The KL gate is not mean-only.** `kl_gate_ok` also caps the **argmax-disagreement
   rate** (default 1%) and an opt-in **p99 KL** — so a *sparse* cheat (bit-exact
   almost everywhere, a few tokens flipped) that keeps `mean_kl` under the threshold
   is still caught by the magnitude-independent flip rate. Calibrate the rate to the
   noise floor: in deterministic mode a faithful kernel sits at **0 flips**, so the
   default is safe; in advisory mode (big MoE) all KL checks are off and accuracy
   carries quality.
6. **Attention has a higher intrinsic KL floor than elementwise ops** (measured on
   the decode-attention swap). A faithful decode kernel — *any* reference SDPA — sits
   at **~6e-3 mean KL vs fa3's flash attention** (flash's online-softmax reduction
   rounds differently, and it compounds over layers), stable across kernel precisions
   and backends. So the **default 5e-3 gate (tuned for silu/rmsnorm) is too strict for
   attention** — the slot needs its own calibrated threshold (~k×6e-3). A broken
   decode kernel sits at **0.126 (20× higher)** and is caught either way; the floor is
   real, not a bug (op-correctness is exact). Per-slot KL thresholds are the fix.

## Repo layout

```
optima/
  slots.py                  # the slot ABI: SlotSpec catalog (7 slots; kind = op|block|collective)
  manifest.py               # bundle manifest parse + path-safety
  sandbox.py                # static policy scan + isolated load (defense-in-depth)
  registry.py               # kernel registry + eligibility + active toggle
  dispatch.py               # per-slot dispatchers — silu/rmsnorm/attention/moe/all_reduce
  verify.py                 # op/block correctness vs HP reference (allclose|matched_ratio|cosine)
  verify_collective.py      # DISTRIBUTED verify for collective slots (mp-spawn N ranks)
  rebuild.py                # fenced escape hatch: validator-shipped repo patchers only (no bundle code)
  compat.py                 # PINNED_SGLANG (0.5.12.post1) + the seam canary (`optima compat`)
  seam.py / bootstrap.py    # install the seam in every venv interpreter via a .pth
  integrations/
    sglang_silu.py / sglang_norm.py        # ops: SiluAndMul, RMSNorm
    sglang_attention.py / sglang_moe.py    # blocks: RadixAttention.forward, FusedMoE.forward
    sglang_allreduce.py                    # collective: GroupCoordinator.all_reduce
    sglang_plugin.py                       # entry point for sglang builds that have a plugin fw
  eval/
    throughput_kl.py        # two-launch throughput + KL (generic corpus; calibration smoke)
    capability.py           # two-launch throughput + KL + benchmark accuracy (the real-task scoring path)
    benchmarks.py           # Benchmark protocol + GSM8K & MMLU (HF), answer extraction
    kl.py / prompts.py / _launch.py
  bundle_hash.py            # deterministic bundle identity
  commit_reveal.py          # commit-reveal + king-of-the-hill ledger
  cli.py                    # slots|compat|scan|verify|evaluate|bench|hash|commit|reveal|ledger|settle
examples/
  miner_silu_{triton,torch,broken,sparse}/     # silu slot (faithful / CPU dry-run / adversarial / sparse)
  miner_rmsnorm_{triton,broken}/               # rmsnorm slot (faithful / adversarial)
  miner_attention_torch/ miner_attention_decode_torch/   # attention.sdpa / attention.decode (blocks)
  miner_moe_fused_experts_torch/               # moe.fused_experts (block)
  miner_moe_mxfp4_sm120/                       # moe.fused_experts_mxfp4 — the sm120 throughput win
  miner_allreduce_torch/                       # collective.all_reduce
tests/                                  # 75 tests (scanner, manifest, KL, verify, block/moe seams, collective, rebuild, commit-reveal)
```

## How a kernel gets into the model (the seam)

`sglang.Engine` forces `mp.set_start_method("spawn")` and runs the model in a
separate scheduler process, so a class-patch in the parent never reaches it. We
install the seam in **every** venv interpreter via a `.pth` file
(`import optima.bootstrap`) + a post-import hook that patches the target chokepoint the
moment its module loads — including in the spawned scheduler. Five chokepoints today:
`SiluAndMul` / `RMSNorm` (ops), `RadixAttention.forward` / `FusedMoE.forward` (blocks),
and `GroupCoordinator.all_reduce` (collective). The pinned sglang (0.5.12.post1, see
`optima/compat.py`) has no stable plugin framework, so this `.pth` path is primary; the
entry-point plugin is kept for builds that do.

The validator does **two launches** of the same model (identical weights/seed):
baseline (`OPTIMA_ACTIVE=0`, stock kernels) and candidate (`OPTIMA_ACTIVE=1`,
miner kernel). Only the one op differs, so the throughput delta and the KL/accuracy
deltas are attributable to the kernel.

**Tamper-resistant timing:** the driver/timer process calls `seam.mark_driver()`
*before* importing sglang, so it never imports miner code; the kernel runs only in
the spawned scheduler, which the driver times over IPC. A malicious kernel can't
reach the clock.

## Run it

### CPU dry-run (no GPU)

```bash
pip install -e .
python -m optima.cli slots
python -m optima.cli verify examples/miner_silu_torch --device cpu --dtype float32
pytest tests/
```

### GPU (the recipe validated on an H100)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python sglang -e . ninja datasets
SP=$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')
echo 'import optima.bootstrap' > "$SP/optima.pth"     # install the seam everywhere
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PWD/.venv/bin:$PATH   # sglang JIT needs nvcc+ninja
export TORCH_CUDA_ARCH_LIST=9.0                        # 9.0=H100, 12.0=RTX Blackwell

# op-correctness on device
.venv/bin/python -m optima.cli verify examples/miner_rmsnorm_triton --device cuda

# cheap KL smoke on a generic corpus (calibration / quick check)
.venv/bin/python -m optima.cli evaluate examples/miner_silu_triton \
    --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic            # KL gate

# the real scoring path: throughput on real benchmark prompts (GSM8K + MMLU, long
# CoT generation), gated on KL *and* task accuracy from the same run
.venv/bin/python -m optima.cli bench examples/miner_silu_triton \
    --model Qwen/Qwen2.5-1.5B-Instruct --benchmarks gsm8k,mmlu --samples 64

# gpt-oss-120b TP=4 on Blackwell (sm_120a): plain-triton MoE, custom-allreduce off
.venv/bin/python -m optima.cli bench examples/miner_rmsnorm_broken \
    --model openai/gpt-oss-120b --benchmarks gsm8k,mmlu --samples 16 \
    --tp-size 4 --moe-runner-backend triton --disable-custom-all-reduce \
    --mem-fraction 0.85   # must FAIL (accuracy collapse and/or KL blowup)
```

## The submission ABI

A bundle is a directory: `manifest.toml` (data — which slots, where the source is)
+ kernel **source** + optional eligibility `metadata/`. The miner provides only the
slot's `entry` callable; the **validator** allocates outputs, owns the dispatch and
fallback, and does the registration. Adding a slot is a validator action in
`optima/slots.py` (+ a seam patch). A slot's `kind` is `op` (one fused op), `block`
(a region of several ops behind one typed boundary), or `collective` (a cross-GPU reduce —
handed the process group, verified distributed). Correctness is `allclose` for bit-faithful
ops, `matched_ratio` (≥ρ of elements within tol vs high-precision ground truth) for kernels
that legitimately differ (attention/fp8/absorbed), or `cosine` (vs the HP reference) for
low-bit kernels where element-wise tolerance is meaningless (MXFP4/MXFP8). **Seven slots
today:**

- `activation.silu_and_mul` — `entry(x, out)` — Qwen/Llama-class MLP (op).
- `norm.rmsnorm` — `entry(x, weight, out, eps)` — universal; fires on gpt-oss (op).
- `attention.sdpa` — `entry(q, k, v, out, sm_scale, causal)` — scaled-dot-product
  attention (block; the op-correctness demo of the wider boundary).
- `attention.decode` — `entry(q, k, v, seq_lens, sm_scale, out)` — paged-decode
  attention; the seam extracts the running model's paged KV and routes decode through
  it (block; eager-only gather MVP — a paged-direct, CUDA-graph-safe contract is next).
- `moe.fused_experts` — `(prepare, forward)` pair — SwiGLU fused experts; `prepare` owns
  the weight layout once at load, `forward(x, topk_ids, topk_weights, prepared, out)` runs
  per step (block).
- `moe.fused_experts_mxfp4` — the MXFP4 variant that **is the sm120 throughput win**:
  `prepare` repacks/interleaves MXFP4 weights+scales, `forward` MXFP8-quantizes and calls
  CUTLASS fused-MoE; gated by `cosine` (block).
- `collective.all_reduce` — `entry(x, out, group)` — TP all-reduce (the comms waist); the
  validator owns the buffer + the process group; verified distributed vs the fp32
  cross-rank sum (`optima.verify_collective`).

## Anti-copy & scoring: commit-reveal + king of the hill

A round is `commit → reveal → evaluate → settle`:

```bash
optima commit  examples/miner_silu_triton --hotkey alice --salt s1 --round 0 --ledger l.json
optima reveal  examples/miner_silu_triton --hotkey alice --salt s1 --round 0 --ledger l.json
optima evaluate examples/miner_silu_triton --model Qwen/Qwen2.5-0.5B-Instruct \
    --ledger l.json --hotkey alice --round 0 --no-deterministic
optima settle  --round 0 --margin 0.02 --ledger l.json
```

- **commit-reveal** binds `H(content_hash, hotkey, salt)`; a reveal must match a
  prior commitment by that hotkey, so you can't reveal a bundle you didn't commit
  to (copying at reveal time is impossible). In production this is Bittensor's
  native commit-reveal (we keep only the off-chain scoring half).
- **copy detection**: earliest commit of a content hash is original; later ones
  earn 0. *(Next: a behavioral/functional fingerprint to catch reformatted
  near-copies — exact hashes miss those; see SUBNET_BLUEPRINT.)*
- **king of the hill**: a champion holds the emission; a challenger takes the title
  only by beating it by a margin. A copy ties → earns nothing.

Robust scoring: median-of-K timed passes with spread, per-epoch seeded prompts
(anti-overfit), and a speedup margin gate.

## Security model

With Triton/CuteDSL the miner's kernel is **Python that runs in the model
process**, so there's no artifact we can prove safe. The boundary must come from
how you run it (and the model is public, so there's no IP to steal):

- the kernel runs on the GPU box, **not** the process that holds chain keys / sets
  weights — those live on a separate CPU control box (Affine's SSH pattern);
- **no network egress** from the GPU box; **ephemeral** per-eval, wiped after;
- a **per-eval CUDA context + watchdog** (DoS / out-of-bounds writes);
- timing is already out-of-process (`mark_driver`); the static scan
  (`sandbox.scan_source`) is a tripwire, not the boundary.

`--framework-mode` and `--isolate` now fail closed if the candidate process
cannot prove no-egress network isolation. Use `--allow-unsafe-no-isolation` only
for local throughput debugging on dev pods that lack `CAP_SYS_ADMIN`; production
scoring should run the eval worker with real namespace support, or inside a
container/VM whose candidate side has `--network=none`.

Worst case for a fully-compromised kernel is one wrong score for itself;
cross-validator consensus catches a rogue validator.

## What's MVP vs. production

| Concern | Now | Production |
|---|---|---|
| Slots | 7: silu/rmsnorm, attention.sdpa/decode, MoE(+MXFP4), all-reduce | + MLA, GEMM, comms-overlap blocks |
| Throughput gain | **first win: sm120 MXFP4 MoE (seam ties the forked backend)** | novel kernels beating mature tuned baselines |
| Model | up to gpt-oss-120b (1 GPU) | DSV4-scale (multi-GPU, TP/PD/EP) |
| Quality gate | KL + GSM8K/MMLU on real prompts, **uncalibrated** | noise-floor KL + large-n benchmarks + det mode |
| Isolation | scan + in-proc load | namespaces + no-egress + per-eval ctx + watchdog |
| Chain | local JSON ledger | on-chain commit-reveal + set_weights |
| State | JSON | a real DB, single-writer weights |

## Adding a slot

1. Define a `SlotSpec` in `optima/slots.py` (`make_inputs`, `invoke_reference`,
   `invoke_entry`, `out_shapes`, a `Correctness` mode, tolerances). It must satisfy the
   four invariants in [docs/SLOT_CONTRACT.md](docs/SLOT_CONTRACT.md); if it can't, it
   belongs in the fenced escape hatch (`rebuild.py`), not the core.
2. Add a seam patch under `optima/integrations/` that routes the real sglang chokepoint
   through a dispatcher built with `make_*_dispatcher`, install it from `seam.activate()`,
   and add the module to `bootstrap._TARGETS`; add a canary line in `optima/compat.py`.
3. Miners target the new slot by name in their manifest. (A `collective` slot is verified
   with `optima.verify_collective`, not `verify_entry` — see the contract doc.)
