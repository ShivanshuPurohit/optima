# Dev environment — GPU pods, toolchain, run recipe

How to get a GPU, push the harness, and run evals. Written so a fresh agent can
pick up cold.

## The toolchain env (`sn120`)

A pyenv on the dev macbook has everything for chain + pods + Affine in one place:

```bash
pyenv activate sn120          # ~/.pyenv/versions/sn120  (Python 3.11)
# has: bittensor 10.x, bittensor-cli, bittensor-wallet, lium-cli 0.6, affine
```

If `pyenv activate` isn't wired into the shell, call binaries directly, e.g.
`~/.pyenv/versions/sn120/bin/lium ...` and `~/.pyenv/versions/sn120/bin/python ...`.

This is also where the **Bittensor SDK** lives for when we build the chain layer
(read commitments, set weights). The Affine subnet code is at
`~/Downloads/github/affine` (studied in `docs/SUBNET_BLUEPRINT.md`).

## The GPU pods (lium.io)

We rent GPUs on [lium](https://github.com/Datura-ai/lium-cli) (Datura-ai). The CLI
is configured (`~/.lium`). **Pod names/IPs change on redeploy — always run
`lium ps` to get the current ones.** As of this writing:

| Pod (name) | GPU | CUDA | Access | Notes |
|---|---|---|---|---|
| `brave-orbit-7c` | **4× RTX PRO 6000 Blackwell** (4×96 GB, GDDR7, no NVLink) | 13.0 | `154.54.100.130` | the **dev box** — TP / PD-disagg / EP / bigger models / Blackwell (FA4, nvfp4) |
| `golden-lion-b6` | H100 (80 GB HBM) | 13.0 | `216.81.245.218:40309` | where the harness was validated |

### Driving a pod programmatically

All from the `sn120` env (`lium` on PATH there):

```bash
lium ps                                   # list pods + names + IPs
lium exec brave-orbit-7c "nvidia-smi -L"  # run a command non-interactively
lium ssh  brave-orbit-7c                  # interactive shell
lium rsync brave-orbit-7c ./optima        # push a directory (use for the harness)
lium scp  brave-orbit-7c ./file /root/    # copy a single file
lium logs brave-orbit-7c                  # stream logs
lium rm   brave-orbit-7c                  # TERMINATE (stops billing)
```

The H100 also answers direct ssh: `ssh root@216.81.245.218 -p 40309 -i ~/.ssh/id_ed25519`.

> Billing: the Blackwell box is ~$4.64/h, the H100 ~$1.48/h. **Tear down idle pods
> with `lium rm <name>`** — they bill while RUNNING.

## Bootstrapping the harness on a fresh pod

The validated recipe (from the H100; adjust `TORCH_CUDA_ARCH_LIST` per GPU):

```bash
# on your machine: push the harness (NOT the sglang clone)
lium rsync brave-orbit-7c ~/Downloads/github/optima/optima
lium rsync brave-orbit-7c ~/Downloads/github/optima/examples

# on the pod (lium ssh, or wrap each in `lium exec brave-orbit-7c "..."`):
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root/optima && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python sglang ninja datasets -e .
SP=$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')
echo 'import optima.bootstrap' > "$SP/optima.pth"     # install the seam everywhere

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PWD/.venv/bin:$PATH   # sglang JIT needs nvcc + ninja
export TORCH_CUDA_ARCH_LIST=12.0                       # 12.0 = RTX PRO 6000 Blackwell (sm_120)
                                                       # 9.0 = H100, 10.0 = B200
.venv/bin/python -m optima.cli verify   examples/miner_silu_triton --device cuda
.venv/bin/python -m optima.cli evaluate examples/miner_silu_triton --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic
```

### Blackwell (sm_120) caveats to verify on first run

- sglang / sgl-kernel / flashinfer Blackwell (sm_120) support is **newer** than
  Hopper — expect possible build/runtime friction; `uv` may resolve a
  Blackwell-capable sgl-kernel/torch. If `pip install sglang` pulls a Hopper-only
  build, you may need a CUDA-13 / Blackwell wheel or a source build.
- nvfp4/mxfp4 is the *optimized* path on Blackwell (native FP4), unlike Hopper
  where we saw "mxfp4 not fully optimized" — quant kernels should be *better* here.
- For multi-GPU work (the reason for this box): `--tp-size 4` etc. Throughput will
  be PCIe-comms-bound (no NVLink) — that's expected; the point is exploring the
  multi-GPU optimization surface (TP/PD/EP), not peak throughput.

## Clean-signal eval settings

When measuring kernel fidelity, run with `enable_deterministic_inference` so the
nondeterminism noise floor → ~0, and calibrate the KL threshold to a measured
stock-vs-stock floor (see `README.md` "Calibration findings"). Measure the floor
with `optima/.../noise_floor`-style stock-vs-stock runs before trusting a KL number.
